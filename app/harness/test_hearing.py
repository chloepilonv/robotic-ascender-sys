"""Headless evidence for the robot's ears. Run it; read the tables.

    ../.venv_everest/bin/python -m app.harness.test_hearing
    ../.venv_everest/bin/python -m app.harness.test_hearing --sections 1 2
    ../.venv_everest/bin/python -m app.harness.test_hearing --clip-stride 6

Five things get measured, because five things could be wrong and "it heard me
when I shouted at my laptop" is not evidence:

  1  STOP DETECTION AND FALSE STOPS, over the whole `say` corpus, at every
     combination of wind 0/6/12/20 m/s and distance 2/5/10 m. Both halves
     matter and the second one matters more: a robot that stops when nobody
     said stop is worse than one that misses a stop, because the demo then
     cannot be driven at all. The false-stop denominator is the NEAR-MISS set
     ("top", "shop", "drop") plus the ordinary calls, and the near-misses are
     the only ones that make the number honest. The section ENDS by sweeping
     the threshold across the pooled results and CHOOSING one.
  2  VOICE ACTIVITY on the same grid: does the VAD open a segment at all? This
     is the gate everything else sits behind -- an utterance that never becomes
     a segment is never given to the recogniser or the correlator, so table 1's
     misses at 10 m in a gale are mostly table 2's misses.
  3  BEARING ERROR vs wind and distance, with the speaker at 0/45/90/135/180
     degrees around the robot. Mean and 95th percentile, because a mic array's
     failure mode is a small number of catastrophically wrong answers, not a
     gentle spread -- and the 95th is the column that shows it.
  4  END TO END, IN THE SIMULATOR. Visibility 3 m (the eyes are blind), the
     hiker 6 m away at 45 degrees, one shout of "come here": does the robot
     turn and walk to her, and how long does it take? Five seeds, plus a clear
     -weather control arm. Then a "stop" mid-walk, and how many ticks until the
     command is zero.
  5  PHYSICS PARITY, the `test_guide` convention: the same scripted command
     flown twice, hearing off and hearing on with the ear model running and the
     detectors firing. The robot's state must come back BIT-identical, because
     the ears are a sensor and a sensor may not move the robot.

WHAT SECTIONS 1-3 DO NOT USE: MuJoCo. They drive `hearing.Ears` directly with a
scripted geometry, because the question "can this array hear that word at that
range in that wind" has nothing to do with what the robot is standing on, and
answering it inside a physics loop would take twenty times as long and mix two
sources of variation. Section 4 is the one that flies.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

from app.harness import hearing as hearing_module
from app.harness import hearing_corpus as corpus_module

# The grid. Winds are the wind dial in m/s; distances are mouth-to-array range.
WIND_SPEEDS_MPS = (0.0, 6.0, 12.0, 20.0)
DISTANCES_METERS = (2.0, 5.0, 10.0)
# THE USER'S FIVE ANGLES, PLUS THREE THAT ARE NOT MULTIPLES OF 45. The first
# version of this table used 0/45/90/135/180 alone and reported an error of
# 0.0 degrees in every single cell -- which is not an array that is perfect, it
# is a table that cannot see. At those five angles the two pairs' delays are
# either equal in magnitude or one of them is zero, so `atan2(u_y, u_x)` gives
# the exact answer for ANY common scaling of the two components: a calibration
# error of 20% in the baseline, the speed of sound or the sample rate would
# have gone straight through. 25, 70 and 160 degrees are where a scaling error
# actually shows up, and they are here for that reason.
BEARING_AZIMUTHS_DEGREES = (0.0, 25.0, 45.0, 70.0, 90.0, 135.0, 160.0, 180.0)
# Candidate thresholds for the sweep in section 1. Vosk's grammar posteriors
# pile up near 1.0, so the interesting range is the top of the scale.
THRESHOLD_LADDER = (0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.00)
# How many missed stops one false stop is worth. See `print_stop_tables`.
FALSE_STOP_COST = 3.0
# HOW LOUDLY SHE SPEAKS, in dB about the corpus's own level. The user's point:
# a shout carries further in wind than a mumble, and nothing in the pipeline
# normalises the microphone, so the source level has to be a variable in the
# tables rather than an assumption.
SOURCE_LEVEL_DECIBELS = (-20.0, 0.0, +10.0)
SOURCE_LEVEL_DISTANCE_METERS = 5.0
SOURCE_LEVEL_WIND_MPS = 12.0
# THE LEAD-IN IS 1.5 s, NOT 0.3, AND IT IS WIND -- not silence. webrtcvad is
# STATEFUL: it adapts an internal noise model, and it needs a second or two of
# the current conditions before it can tell a voice from a gale. That makes
# every table order-dependent unless each clip is given its own warm-up, and it
# bit hard: the demo clips read "never heard" at every wind above zero, while
# the same clips inside the big sweep read 100% -- because in the sweep 365
# clips at one wind speed had already adapted the detector, and in the small
# table the wind changed every three clips. Through the ear model a lead-in of
# zero SOURCE is a lead-in of pure wind noise, which is exactly the warm-up the
# detector needs, so the fix is one constant and the tables stop depending on
# the order they were computed in.
LEADING_SILENCE_SECONDS = 1.50
TRAILING_SILENCE_SECONDS = 0.50   # ... and enough after it to close the segment
CONTROL_HZ = 50.0
SAMPLES_PER_TICK = int(round(hearing_module.SAMPLE_RATE_HZ / CONTROL_HZ))

# Section 4.
# WHICH WORLD SECTION 4 FLIES, and it is a MEASURED choice -- table 4a below.
# `flat_free` (added 2026-08-30: perfectly flat, zero roughness, no rope) is the
# default because it removes the terrain as a variable entirely; the ladder
# below still measures the alternatives so the choice stays evidenced.
# The ear layer's only body actuator is `ang_vel_yaw`, so it needs a world where
# a commanded turn produces a turn AND the robot stays upright long enough to
# walk six metres. On the roped worlds a turn does nothing (PARITY.md: the palm
# is clipped to a fixed line). On `sandbox_free` the yaw responds but the walker
# TIPS OVER inside four seconds under a walk-and-turn command. `terrain_free_0`
# -- the flat Lhotse-roughness ground, rope off, added 2026-08-30 -- is the one
# world in the catalogue where the robot neither falls nor is deaf to the
# command, and it is what section 4 flies.
DEFAULT_SIM_WORLD = "flat_free"
SIM_CALL_DISTANCE_METERS = 6.0
SIM_CALL_AZIMUTH_DEGREES = 45.0
SIM_WHITEOUT_VISIBILITY_METERS = 3.0
SIM_CLEAR_VISIBILITY_METERS = 100.0
# A LADDER, not one setting. At 3 m the eyes cannot see a person 6 m away at
# all, so "does the robot reach WAIT" cannot be answered there by construction
# -- it is the arm that proves the EARS are the only sensor in play. 10 m and
# 100 m are what show the hand-over back to vision actually happening.
SIM_VISIBILITY_LADDER_METERS = (3.0, 10.0, 100.0)
SIM_ARRIVAL_METERS = 1.3          # the follower's own FOLLOW/WAIT boundary
SIM_SECONDS = 90.0
SIM_CALL_AT_SECONDS = 1.0
SIM_STOP_AT_SECONDS = 12.0
# The eye cameras' horizontal half field of view, from `guide.StereoEyes`
# (320x240 at the d435i's fovy). Used only to ask whether the ear cue put her
# in front of the cameras.
CAMERA_HALF_FOV_DEGREES = 36.5


# ------------------------------------------------------ the offline ear bench
def hear_clip(clip, distance_meters, azimuth_degrees, wind_speed, seed=0,
              ears=None):
    """One utterance, through the ear model, into the real detectors.

    The geometry is scripted: the robot's array sits at the origin with its
    body axes aligned to the world, and the mouth sits at `distance` metres and
    `azimuth` degrees around it (+ = the robot's LEFT). The clip is padded with
    silence either side, then pushed through `Ears.update` one CONTROL TICK at a
    time -- the same 320-sample slices the live loop uses, through the same
    segment machinery -- so what this measures is the shipped path, not a
    convenient reimplementation of it.

    Output: a list of segment verdicts,
    `{heard, stop_confidence, bearing_degrees, bearing_confidence,
      voice_probability}`, one per utterance the VAD found. Empty means the
    voice was never heard at all.
    """
    if ears is None:
        ears = hearing_module.Ears(seed=seed, verbose=False,
                                   control_hz=CONTROL_HZ)
    else:
        ears.reset()
    radians = math.radians(azimuth_degrees)
    mouth = np.array([distance_meters * math.cos(radians),
                      distance_meters * math.sin(radians), 0.0])
    head_position = np.zeros(3)
    head_rotation = np.eye(3)

    padded = np.concatenate((
        np.zeros(int(LEADING_SILENCE_SECONDS * hearing_module.SAMPLE_RATE_HZ),
                 dtype=np.float32),
        np.asarray(clip, dtype=np.float32),
        np.zeros(int(TRAILING_SILENCE_SECONDS * hearing_module.SAMPLE_RATE_HZ),
                 dtype=np.float32)))
    segments = []
    ticks = int(math.ceil(padded.size / SAMPLES_PER_TICK))
    for tick in range(ticks):
        chunk = padded[tick * SAMPLES_PER_TICK:(tick + 1) * SAMPLES_PER_TICK]
        if chunk.size < SAMPLES_PER_TICK:
            chunk = np.concatenate((chunk, np.zeros(
                SAMPLES_PER_TICK - chunk.size, dtype=np.float32)))
        ears.update(chunk, mouth, head_position, head_rotation, wind_speed, tick)
        if ears.new_segment:
            segments.append({
                "heard": ears.heard,
                "stop_confidence": float(ears.stop_confidence),
                "bearing_degrees": (None if ears.bearing_radians is None else
                                    math.degrees(ears.bearing_radians)),
                "bearing_confidence": float(ears.bearing_confidence),
                "voice_probability": float(ears.segment_peak_voice_probability),
            })
    return segments


def sweep_corpus(manifest, clips, ears, distances=DISTANCES_METERS,
                 winds=WIND_SPEEDS_MPS, azimuth_degrees=0.0, seed=0):
    """Every clip at every (wind, distance). -> {(wind, distance): [rows]}.

    ONE PASS SERVES TABLES 1 AND 2, because they are two questions about the
    same event: the VAD decides whether there is a segment, and only then does
    the recogniser see anything. Running them separately would double the cost
    and let the two tables disagree about which clips were heard.
    """
    results = {}
    total = len(winds) * len(distances) * len(clips)
    done, started = 0, time.time()
    for wind in winds:
        for distance in distances:
            rows = []
            for row in clips:
                clip = corpus_module.read_clip(manifest, row)
                segments = hear_clip(clip, distance, azimuth_degrees, wind,
                                     seed=seed, ears=ears)
                best = max((segment["stop_confidence"] for segment in segments),
                           default=0.0)
                rows.append({
                    "label": row["label"],
                    "text": row["text"],
                    "voice": row["voice"],
                    "segments": len(segments),
                    "stop_confidence": best,
                    "voice_probability": max(
                        (segment["voice_probability"] for segment in segments),
                        default=0.0),
                })
                done += 1
            results[(wind, distance)] = rows
            elapsed = time.time() - started
            print(f"  ... wind {wind:4.0f} m/s, {distance:4.1f} m:"
                  f" {len(rows)} clips  [{done}/{total},"
                  f" {elapsed:.0f} s elapsed,"
                  f" {elapsed / max(done, 1) * (total - done):.0f} s left]",
                  flush=True)
    return results


# ------------------------------------------------------------- 1: the word
def print_stop_tables(results) -> float:
    """Tables 1a/1b/1c, and the CHOSEN threshold. -> the threshold."""
    print("\n1. THE WORD `stop`.  Every corpus clip at every (wind, distance),"
          " speaker dead ahead.")
    print("   `detected` = the best `stop` confidence over the utterance's"
          " segments reached the threshold.")

    def rate(rows, labels, threshold):
        selected = [row for row in rows if row["label"] in labels]
        if not selected:
            return float("nan"), 0
        fired = sum(1 for row in selected
                    if row["stop_confidence"] >= threshold)
        return fired / len(selected), len(selected)

    print("\n1a. THRESHOLD SWEEP, pooled over the whole grid"
          " (the number below is chosen from this table)")
    print(f"| threshold | stop detected | false stop, near-miss |"
          f" false stop, other calls | false stop, pooled |"
          f" detected - {FALSE_STOP_COST:.0f}x false |")
    print("|---|---|---|---|---|---|")
    pooled = [row for rows in results.values() for row in rows]
    best_threshold, best_margin = THRESHOLD_LADDER[0], -2.0
    for threshold in THRESHOLD_LADDER:
        hit, _ = rate(pooled, ("stop",), threshold)
        near, _ = rate(pooled, ("near_miss",), threshold)
        other, _ = rate(pooled, ("other",), threshold)
        false_pooled, _ = rate(pooled, corpus_module.NON_STOP_LABELS, threshold)
        # THE CHOICE RULE, stated before the table is read: maximise
        # `detection - FALSE_STOP_COST * false-stop`. The weight is not a
        # default, it is the brief: the user named "top", "shop" and "drop" as
        # near-misses that MUST NOT trigger, so a false stop is declared to cost
        # three misses. Writing it here means the threshold is chosen by a rule
        # rather than by looking at the numbers and picking a nice one.
        margin = hit - FALSE_STOP_COST * false_pooled
        if margin > best_margin:
            best_threshold, best_margin = threshold, margin
        print(f"| {threshold:.2f} | {100 * hit:5.1f}% | {100 * near:5.1f}% |"
              f" {100 * other:5.1f}% | {100 * false_pooled:5.1f}% |"
              f" {100 * margin:+6.1f} |")
    print(f"\n  CHOSEN THRESHOLD {best_threshold:.2f}"
          f"  (rule: maximise detection - false-stop, declared above;"
          f" margin {100 * best_margin:+.1f} points)")
    print(f"  `hearing.STOP_CONFIDENCE_THRESHOLD` is currently"
          f" {hearing_module.STOP_CONFIDENCE_THRESHOLD:.2f}"
          + ("  -- AGREES" if abs(best_threshold
                                  - hearing_module.STOP_CONFIDENCE_THRESHOLD) < 1e-9
             else "  -- DISAGREES: move the constant to the chosen value"))

    for title, labels, key in (
            ("1b. STOP DETECTION RATE", ("stop",), "detect"),
            ("1c. FALSE-STOP RATE, near-misses + ordinary calls",
             corpus_module.NON_STOP_LABELS, "false")):
        print(f"\n{title}  (threshold {best_threshold:.2f})")
        print("| wind m/s | " + " | ".join(f"{d:.0f} m" for d in DISTANCES_METERS)
              + " |")
        print("|---" * (len(DISTANCES_METERS) + 1) + "|")
        for wind in WIND_SPEEDS_MPS:
            cells = []
            for distance in DISTANCES_METERS:
                rows = results.get((wind, distance), [])
                value, count = rate(rows, labels, best_threshold)
                cells.append(f"{100 * value:5.1f}% ({count})")
            print(f"| {wind:.0f} | " + " | ".join(cells) + " |")
    return best_threshold


def print_demo_clip_table(manifest, ears, threshold, seed=0) -> None:
    """1e: the five clips the SIDEBAR actually plays, each on the whole grid.

    They are in the pooled corpus as well, where five rows among 365 cannot
    move a number. This table is here because they are the utterances the demo
    will be judged on: a threshold tuned on `say` voices that misses the very
    clip the PEMBA button plays is a threshold tuned on the wrong thing.
    """
    demo = [row for row in manifest["clips"] if row["voice"] == "demo"]
    if not demo:
        print("\n1e. THE DEMO CLIPS -- none found in"
              f" {corpus_module.DEMO_VOICE_DIRECTORY}")
        return
    print(f"\n1e. THE DEMO CLIPS -- app/web/sounds/voice/, what the sidebar's"
          f" MANUAL buttons play.  `stop` threshold {threshold:.2f}.")
    print("| clip | label | seconds | " + " | ".join(
        f"{wind:.0f} m/s" for wind in WIND_SPEEDS_MPS) + " |")
    print("|---" * (len(WIND_SPEEDS_MPS) + 3) + "|")
    for row in demo:
        clip = corpus_module.read_clip(manifest, row)
        cells = []
        for wind in WIND_SPEEDS_MPS:
            verdicts = []
            for distance in DISTANCES_METERS:
                segments = hear_clip(clip, distance, 45.0, wind, seed=seed,
                                     ears=ears)
                best = max((segment["stop_confidence"]
                            for segment in segments), default=0.0)
                if not segments:
                    verdicts.append("·")           # never heard at all
                elif best >= threshold:
                    verdicts.append("S")           # heard, decoded as `stop`
                else:
                    verdicts.append("v")           # heard, called a voice
            cells.append("".join(verdicts))
        print(f"| `{row['path']}` | {row['label']} | {row['seconds']:.2f} | "
              + " | ".join(cells) + " |")
    print("   Each cell is 2 m / 5 m / 10 m: `S` decoded as `stop`, `v` heard"
          " as an ordinary voice, `·` not heard at all. A `stop` row wants"
          " SSS; every other row wants vvv and MUST NOT contain an S.")


def print_source_level_table(manifest, clips, ears, seed=0) -> None:
    """1d: does speaking up help? It had better.

    NOTHING IN THIS PIPELINE APPLIES AUTOMATIC GAIN (user's ruling,
    2026-08-30). The page asks the browser for `autoGainControl: false`, the
    runtime does not normalise the PCM it receives, and the ear model applies
    only the world -- 1/r, the propagation delay, the air's low-pass and the
    wind noise. The consequence is testable and is tested here: at a fixed
    range in a fixed wind, a louder utterance must be heard more often, decoded
    more often, and located more accurately. If this table were flat, something
    upstream would be quietly normalising and the whole "shout at the robot"
    story would be a lie.
    """
    print(f"\n1d. SOURCE LEVEL -- the same clips at"
          f" {SOURCE_LEVEL_DISTANCE_METERS:.0f} m in"
          f" {SOURCE_LEVEL_WIND_MPS:.0f} m/s of wind, spoken at three"
          f" loudnesses.")
    print("| source level | peak ear level | SNR vs wind | voice heard |"
          " `stop` decoded | bearing mean err |")
    print("|---|---|---|---|---|---|")
    noise = hearing_module.wind_noise_amplitude(SOURCE_LEVEL_WIND_MPS)
    for level_db in SOURCE_LEVEL_DECIBELS:
        gain = 10.0 ** (level_db / 20.0)
        heard = decoded = stop_clips = 0
        errors, ear_levels = [], []
        for row in clips:
            clip = corpus_module.read_clip(manifest, row) * gain
            segments = hear_clip(clip, SOURCE_LEVEL_DISTANCE_METERS, 45.0,
                                 SOURCE_LEVEL_WIND_MPS, seed=seed, ears=ears)
            if not segments:
                continue
            heard += 1
            ear_levels.append(ears.segment_peak_level_db)
            measured = segments[0]["bearing_degrees"]
            if measured is not None:
                errors.append(abs((measured - 45.0 + 180) % 360 - 180))
            if row["label"] == "stop":
                stop_clips += 1
                best = max(segment["stop_confidence"] for segment in segments)
                if best >= hearing_module.STOP_CONFIDENCE_THRESHOLD:
                    decoded += 1
        source_rms = (hearing_module.VOICE_REFERENCE_RMS_AT_ONE_METER * gain
                      / SOURCE_LEVEL_DISTANCE_METERS)
        print(f"| {level_db:+.0f} dB |"
              f" {np.mean(ear_levels) if ear_levels else float('nan'):.1f} dBFS |"
              f" {20 * math.log10(source_rms / noise):+.1f} dB |"
              f" {100 * heard / max(len(clips), 1):5.1f}% |"
              f" {100 * decoded / max(stop_clips, 1):5.1f}% ({stop_clips}) |"
              f" {np.mean(errors) if errors else float('nan'):5.1f}° |")
    print("   The stop-clip count is small at this stride; what the column is"
          " for is the SHAPE, and the shape is the claim.")


# ------------------------------------------------------------- 2: the voice
def print_voice_table(results) -> None:
    print("\n2. VOICE ACTIVITY (webrtcvad).  Did the utterance become a SEGMENT"
          " at all?")
    print("   Nothing downstream sees an utterance the VAD missed, so this is"
          " the ceiling on table 1.")
    print("| wind m/s | " + " | ".join(f"{d:.0f} m" for d in DISTANCES_METERS)
          + " |")
    print("|---" * (len(DISTANCES_METERS) + 1) + "|")
    for wind in WIND_SPEEDS_MPS:
        cells = []
        for distance in DISTANCES_METERS:
            rows = results.get((wind, distance), [])
            if not rows:
                cells.append("--")
                continue
            heard = sum(1 for row in rows if row["segments"] > 0)
            share = np.mean([row["voice_probability"] for row in rows])
            cells.append(f"{100 * heard / len(rows):5.1f}%  (p {share:.2f})")
        print(f"| {wind:.0f} | " + " | ".join(cells) + " |")
    print("   (p is the peak of the rolling 480 ms voiced-frame share during"
          " the utterance -- the number the HUD's meter shows)")


def print_aggressiveness_table(manifest, clips, seed=0) -> None:
    """2b: why `VOICE_ACTIVITY_AGGRESSIVENESS` is 2 and not 0, 1 or 3."""
    print("\n2b. VAD AGGRESSIVENESS LADDER -- webrtcvad's one knob, swept."
          "  Segment-opened rate.")
    print("   Higher is STRICTER. The ladder is not monotone in wind, which is"
          " worth knowing: 3 is the only setting that hears anything at all in"
          " a 20 m/s gale, and it is also the worst at 12 m/s and 10 m (4.2%),"
          " because a strict detector throws away a quiet voice as readily as"
          " it throws away noise.")
    print("| aggressiveness | wind m/s | "
          + " | ".join(f"{d:.0f} m" for d in DISTANCES_METERS) + " |")
    print("|---" * (len(DISTANCES_METERS) + 2) + "|")
    for aggressiveness in (0, 1, 2, 3):
        ears = hearing_module.Ears(seed=seed, verbose=False,
                                   control_hz=CONTROL_HZ)
        ears.voice_activity = hearing_module.VoiceActivity(aggressiveness)
        for wind in WIND_SPEEDS_MPS:
            cells = []
            for distance in DISTANCES_METERS:
                heard = 0
                for row in clips:
                    clip = corpus_module.read_clip(manifest, row)
                    if hear_clip(clip, distance, 0.0, wind, ears=ears):
                        heard += 1
                cells.append(f"{100 * heard / max(len(clips), 1):5.1f}%")
            mark = ("  <- shipped"
                    if aggressiveness == hearing_module.VOICE_ACTIVITY_AGGRESSIVENESS
                    and wind == WIND_SPEEDS_MPS[0] else "")
            print(f"| {aggressiveness}{mark} | {wind:.0f} | "
                  + " | ".join(cells) + " |")


# ----------------------------------------------------------- 3: the bearing
def bearing_table(manifest, clips, ears, seed=0) -> list:
    """Section 3: azimuth error over the grid. -> rows."""
    rows = []
    total = (len(WIND_SPEEDS_MPS) * len(DISTANCES_METERS)
             * len(BEARING_AZIMUTHS_DEGREES) * len(clips))
    done, started = 0, time.time()
    for wind in WIND_SPEEDS_MPS:
        for distance in DISTANCES_METERS:
            for azimuth in BEARING_AZIMUTHS_DEGREES:
                errors, confidences, heard = [], [], 0
                for row in clips:
                    clip = corpus_module.read_clip(manifest, row)
                    segments = hear_clip(clip, distance, azimuth, wind,
                                         seed=seed, ears=ears)
                    done += 1
                    measured = next((segment["bearing_degrees"]
                                     for segment in segments
                                     if segment["bearing_degrees"] is not None),
                                    None)
                    if measured is None:
                        continue
                    heard += 1
                    errors.append(abs((measured - azimuth + 180) % 360 - 180))
                    confidences.append(segments[0]["bearing_confidence"])
                rows.append({
                    "wind": wind, "distance": distance, "azimuth": azimuth,
                    "heard": heard, "clips": len(clips),
                    "mean_error": float(np.mean(errors)) if errors else float("nan"),
                    "p95_error": (float(np.percentile(errors, 95)) if errors
                                  else float("nan")),
                    "confidence": (float(np.mean(confidences)) if confidences
                                   else float("nan")),
                })
            elapsed = time.time() - started
            print(f"  ... wind {wind:4.0f} m/s, {distance:4.1f} m"
                  f"  [{done}/{total}, {elapsed:.0f} s elapsed,"
                  f" {elapsed / max(done, 1) * (total - done):.0f} s left]",
                  flush=True)
    return rows


def print_bearing_table(rows) -> None:
    print("\n3. BEARING, by GCC-PHAT across the front/back and left/right pairs.")
    print("   Error is |measured - true| in degrees, over the clips the VAD"
          " heard; `--` means the utterance was never heard at that cell, and a"
          " bearing you never got is not a bearing you got wrong.")
    print("| wind m/s | distance | "
          + " | ".join(f"{a:.0f}°" for a in BEARING_AZIMUTHS_DEGREES) + " |")
    print("|---" * (len(BEARING_AZIMUTHS_DEGREES) + 2) + "|")
    for wind in WIND_SPEEDS_MPS:
        for distance in DISTANCES_METERS:
            cells = []
            for azimuth in BEARING_AZIMUTHS_DEGREES:
                row = next((r for r in rows if r["wind"] == wind
                            and r["distance"] == distance
                            and r["azimuth"] == azimuth), None)
                if row is None or row["heard"] == 0:
                    cells.append("     --     ")
                    continue
                cells.append(f"{row['mean_error']:5.1f} /"
                             f" {row['p95_error']:5.1f}")
            print(f"| {wind:.0f} | {distance:.0f} m | " + " | ".join(cells) + " |")
    print("   (each cell is MEAN / 95th percentile, degrees)")
    print("\n3b. HOW OFTEN A BEARING EXISTED AT ALL, and how sharp its peak was")
    print("| wind m/s | distance | heard | mean peak sharpness -> confidence |")
    print("|---|---|---|---|")
    for wind in WIND_SPEEDS_MPS:
        for distance in DISTANCES_METERS:
            selected = [r for r in rows if r["wind"] == wind
                        and r["distance"] == distance]
            heard = sum(r["heard"] for r in selected)
            clips = sum(r["clips"] for r in selected)
            confidences = [r["confidence"] for r in selected
                           if np.isfinite(r["confidence"])]
            confidence = np.mean(confidences) if confidences else float("nan")
            print(f"| {wind:.0f} | {distance:.0f} m |"
                  f" {100 * heard / max(clips, 1):5.1f}% ({heard}/{clips}) |"
                  f" {confidence:.2f} |")


# --------------------------------------------------------- 4: the whole thing
YAW_LADDER_WORLDS = ("flat_free", "terrain_free_0", "flat_0", "terrain_free_5",
                     "sandbox_free")
YAW_LADDER_SECONDS = 4.0


def yaw_authority_table(worlds=YAW_LADDER_WORLDS) -> None:
    """4a: can this robot TURN on this world? The ear layer's one actuator.

    Everything `COMING_BY_EARS` does is command `ang_vel_yaw` toward a
    remembered heading, so a world where a commanded turn does nothing makes
    the whole behaviour untestable -- and this repo already knows that happens:
    PARITY.md records +1.0 and -1.0 rad/s ending 10 degrees apart on the roped
    climb worlds. This is the same measurement on the candidate rope-off
    worlds, and it is why section 4 flies the one it flies.
    """
    from app.harness.runtime import root_yaw_radians

    print("\n4a. YAW AUTHORITY -- a pure yaw command for"
          f" {YAW_LADDER_SECONDS:.0f} s, from the spawn, no other input.")
    print("| world | +1.0 rad/s | 0.0 | -1.0 rad/s | separation | drift at 0 |")
    print("|---|---|---|---|---|---|")
    for world in worlds:
        turns, drift = {}, 0.0
        for rate in (1.0, 0.0, -1.0):
            scene, episode, meta = open_world(world)
            episode.reset()
            start_yaw = root_yaw_radians(episode.data.qpos[3:7])
            start_position = episode.data.qpos[0:2].copy()
            command = np.array([0.0, 0.0, rate])
            for _ in range(int(YAW_LADDER_SECONDS * episode.control_hz)):
                episode.step(command, np.zeros(2))
            end_yaw = root_yaw_radians(episode.data.qpos[3:7])
            turns[rate] = math.degrees(
                (end_yaw - start_yaw + math.pi) % (2 * math.pi) - math.pi)
            if rate == 0.0:
                drift = float(np.linalg.norm(
                    episode.data.qpos[0:2] - start_position))
        separation = turns[1.0] - turns[-1.0]
        mark = "  <- flown" if world == DEFAULT_SIM_WORLD else ""
        print(f"| {world}{mark} | {turns[1.0]:+.0f}° | {turns[0.0]:+.0f}° |"
              f" {turns[-1.0]:+.0f}° | {separation:+.0f}° | {drift:.2f} m |")
    print("   `separation` is the whole test: a robot that cannot put daylight"
          " between +1 and -1 rad/s cannot be steered by anything, ears"
          " included. The drift column is what the walker does with NO command"
          " at all, and it is the noise floor every other number sits on.")


def open_world(name, verbose=False):
    """The same order `runtime.open_world` uses: surgery, sky, look, episode."""
    from app.harness import climb_worlds as climb_worlds_module
    from app.harness import graphics as graphics_module
    from app.harness import guide as guide_module

    library = climb_worlds_module.ClimbSceneLibrary(verbose=verbose)
    scene, meta, definition = library.load(name)
    guide_module.attach_guide(scene, verbose=verbose)
    graphics_module.add_skybox(scene, verbose=verbose)
    graphics_module.apply_alpine_look(
        scene.model, terrain_size_meters=scene.terrain.size_xy)
    episode = climb_worlds_module.ClimbSceneEpisode(
        scene, meta, definition, name, seed=0)
    return scene, episode, meta


def _pin_guide(system, robot_position, robot_yaw, distance, azimuth_degrees):
    """Stand the hiker at a fixed world point, `azimuth` off the robot's nose.

    `Guide` normally walks the rope route with a fixed lateral offset, which is
    the demo. A test needs her at a stated bearing, so her ground position is
    shadowed on the INSTANCE (a plain attribute, which Python looks up before
    the class's method) and nothing in `guide.py` is touched. She still faces
    along the route, still snaps to the terrain height, and still animates.
    """
    heading = robot_yaw + math.radians(azimuth_degrees)
    x = float(robot_position[0]) + distance * math.cos(heading)
    y = float(robot_position[1]) + distance * math.sin(heading)
    terrain = system.guide.terrain
    point = np.array([x, y, float(terrain.surface_z(x, y))])
    system.guide.ground_position_world = lambda: point.copy()
    return point


def true_head_frame_azimuth(hearing, system, data) -> float:
    """The direction to her mouth in the ARRAY's own frame. -> degrees.

    THE ONLY FAIR COMPARISON FOR THE EAR BEARING, and it is not the 45 degrees
    the hiker was placed at. That 45 is a ground-plane angle measured from the
    robot's PELVIS yaw at spawn; the array rides the TORSO, which pitches and
    rolls as the robot walks, and it sits half a metre below her mouth. Scoring
    the ears against the placement angle would charge them for a frame
    difference: MEASURED, the ears read +31 degrees when the placement said +45
    and the head frame said +31.
    """
    head_position, head_rotation = hearing.head_frame(data)
    mouth = hearing.mouth_position_world(system)
    local = head_rotation.T @ (np.asarray(mouth) - head_position)
    return math.degrees(math.atan2(local[1], local[0]))


def end_to_end(world, clip_path, stop_clip_path, seed, visibility_meters,
               seconds=SIM_SECONDS, stop_at_seconds=None, verbose=False,
               voice="?", distance_meters=SIM_CALL_DISTANCE_METERS) -> dict:
    """Section 4: one flight. Call her once by voice; walk to her.

    Everything the live loop does, minus the websocket and the recorder:
    visibility -> the storm's eye degradation, the guide system's vision, the
    hearing system's ears, the behaviour's command, `episode.step`.
    """
    from app.harness import guide as guide_module
    from app.harness import storm as storm_module
    from app.harness.runtime import root_yaw_radians

    scene, episode, meta = open_world(world)
    storm_vision = storm_module.StormVision(seed=seed)
    storm_vision.update(visibility_meters)
    system = guide_module.GuideSystem(scene, scene.model, episode.control_hz,
                                      verbose=verbose,
                                      degradation=storm_vision.degrade)
    hearing = hearing_module.HearingSystem(scene.model, episode.control_hz,
                                           seed=seed, verbose=verbose)
    hearing.add_injector(clip_path, SIM_CALL_AT_SECONDS) if verbose else None
    if not verbose:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            hearing.add_injector(clip_path, SIM_CALL_AT_SECONDS)
            if stop_at_seconds is not None:
                hearing.add_injector(stop_clip_path, stop_at_seconds)
    elif stop_at_seconds is not None:
        hearing.add_injector(stop_clip_path, stop_at_seconds)

    # THE "NECK" HOOK, exactly as `runtime.make_guide` registers it. Without it
    # the waist offset the ear layer writes goes nowhere and the cameras never
    # turn toward the shout -- a test that quietly measured a feature it had
    # not wired up.
    episode.control_hooks.append(system.waist.apply)
    episode.reset()
    system.place(episode.spawn_position_world)
    robot_yaw = root_yaw_radians(episode.data.qpos[3:7])
    her = _pin_guide(system, episode.spawn_position_world, robot_yaw,
                     distance_meters, SIM_CALL_AZIMUTH_DEGREES)

    ticks = int(seconds * episode.control_hz)
    samples, transitions = [], []
    previous_mode = None
    arrived_at = None
    stopped_at = None
    first_ear_bearing = None
    true_bearing = None
    truth_history = []
    eyes_acquired_at = None
    in_cone_at = None
    for tick in range(ticks):
        time_seconds = tick / episode.control_hz
        guide_command = system.update(episode.data, tick, True, False, False)
        command = hearing.update(episode.data, tick, True, system, guide_command,
                                 0.0, time_seconds)
        if command is None:
            command = guide_command
        episode.step(np.asarray(command, dtype=float), np.zeros(2))
        mode = hearing.behaviour.mode
        if mode != previous_mode:
            transitions.append((time_seconds, previous_mode, mode))
            previous_mode = mode
        # THE TRUTH AZIMUTH IS RECORDED EVERY TICK, because the bearing has to
        # be scored against the geometry AT THE MOMENT ITS AUDIO ARRIVED, not
        # at the moment the verdict is published. Those are up to a second
        # apart (`Ears.bearing_age_seconds`), and this walker swings its torso
        # yaw fast enough that a second is tens of degrees: scoring against the
        # publication tick charged the ears a 43 degree error they had not
        # made.
        truth_history.append((time_seconds,
                              true_head_frame_azimuth(hearing, system,
                                                      episode.data)))
        if (first_ear_bearing is None and hearing.ears.new_segment
                and hearing.behaviour.cue_bearing_radians is not None):
            first_ear_bearing = math.degrees(hearing.behaviour.cue_bearing_radians)
            measured_at = time_seconds - hearing.ears.bearing_age_seconds
            true_bearing = min(truth_history,
                               key=lambda row: abs(row[0] - measured_at))[1]
        if eyes_acquired_at is None and hearing.behaviour.eyes_have_her(
                system.follower):
            eyes_acquired_at = time_seconds
        # DID THE EAR CUE PUT HER IN FRONT OF THE CAMERAS? The eyes' horizontal
        # half-FOV is 36.5 deg about wherever the waist is pointing, so this is
        # the geometric question the ear cue actually has to answer -- and it is
        # separable from whether the DETECTOR then saw her, which the fog and
        # her orange pack decide.
        if in_cone_at is None:
            camera_bearing = (truth_history[-1][1]
                              - math.degrees(system.waist.measured_radians))
            if abs((camera_bearing + 180) % 360 - 180) <= CAMERA_HALF_FOV_DEGREES:
                in_cone_at = time_seconds
        true_range = float(np.linalg.norm(
            episode.data.qpos[0:2] - her[:2]))
        if arrived_at is None and true_range <= SIM_ARRIVAL_METERS:
            arrived_at = time_seconds
        if stopped_at is None and mode == "STOPPED":
            stopped_at = time_seconds
        samples.append({"time_seconds": time_seconds, "mode": mode,
                        "true_range_meters": true_range,
                        "command": np.asarray(command, dtype=float).copy()})
        if arrived_at is not None and stop_at_seconds is None:
            break
        if episode.fell_at_seconds is not None:
            break
        if stopped_at is not None and time_seconds > stopped_at + 1.0:
            break
    system.close()
    return {
        "seed": seed,
        "voice": voice,
        "distance_meters": distance_meters,
        "visibility_meters": visibility_meters,
        "transitions": transitions,
        "arrived_at_seconds": arrived_at,
        "stopped_at_seconds": stopped_at,
        "eyes_acquired_at_seconds": eyes_acquired_at,
        "in_camera_cone_at_seconds": in_cone_at,
        "first_ear_bearing_degrees": first_ear_bearing,
        "true_bearing_degrees": true_bearing,
        "start_range_meters": samples[0]["true_range_meters"],
        "final_range_meters": samples[-1]["true_range_meters"],
        "closest_range_meters": min(s["true_range_meters"] for s in samples),
        "final_mode": samples[-1]["mode"],
        "fell_at_seconds": episode.fell_at_seconds,
        "samples": samples,
    }


def print_end_to_end(rows) -> None:
    print("\n4. END TO END, IN THE SIMULATOR."
          f"  Hiker {SIM_CALL_DISTANCE_METERS:.0f} m away at"
          f" {SIM_CALL_AZIMUTH_DEGREES:.0f}° off the nose, one shout of"
          " \"come here\".")
    print("   `arrived` is graded by the SIMULATOR's own distance -- a"
          " LABELLED CHEAT, used to grade and never in the decision path.")
    print("| visibility | start | seed | voice |"
          " ear bearing vs truth (head frame) |"
          " she enters the camera cone | eyes acquire her | arrived (<= 1.3 m) |"
          " closest | final mode | fell |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        bearing = ("--" if row["first_ear_bearing_degrees"] is None
                   else f"{row['first_ear_bearing_degrees']:+.1f}° vs"
                        f" {row['true_bearing_degrees']:+.1f}° true"
                        f" ({row['first_ear_bearing_degrees'] - row['true_bearing_degrees']:+.1f}°)")
        eyes = ("never" if row["eyes_acquired_at_seconds"] is None
                else f"{row['eyes_acquired_at_seconds']:.1f} s")
        cone = ("never" if row["in_camera_cone_at_seconds"] is None
                else f"{row['in_camera_cone_at_seconds']:.1f} s")
        arrived = ("no" if row["arrived_at_seconds"] is None
                   else f"{row['arrived_at_seconds']:.1f} s")
        print(f"| {row['visibility_meters']:.0f} m |"
              f" {row['distance_meters']:.1f} m | {row['seed']} |"
              f" {row['voice']} | {bearing} | {cone} |"
              f" {eyes} | {arrived} | {row['closest_range_meters']:.2f} m |"
              f" {row['final_mode']} | {row['fell_at_seconds']} |")
    arrived = [row for row in rows if row["arrived_at_seconds"] is not None]
    if arrived:
        times = [row["arrived_at_seconds"] for row in arrived]
        print(f"  {len(arrived)}/{len(rows)} arrived;"
              f" time to arrival mean {np.mean(times):.1f} s,"
              f" min {np.min(times):.1f}, max {np.max(times):.1f}")
    else:
        print(f"  0/{len(rows)} arrived.")
    for row in rows:
        print(f"  seed {row['seed']} @ {row['visibility_meters']:.0f} m: "
              + ", ".join(f"{t:.1f}s {old}->{new}"
                          for t, old, new in row["transitions"]))


def print_stop_mid_walk(row, control_hz=50.0) -> None:
    print(f"\n4b. `stop` MID-WALK. The same flight, with a \"stop\" injected at"
          f" t = {SIM_STOP_AT_SECONDS:.0f} s while the robot is walking.")
    if row["stopped_at_seconds"] is None:
        print("  NOT STOPPED. The utterance was never recognised as `stop`.")
        return
    delay = row["stopped_at_seconds"] - SIM_STOP_AT_SECONDS
    print(f"| injected at | STOPPED at | delay | ticks | command after |")
    print("|---|---|---|---|---|")
    after = [s for s in row["samples"]
             if s["time_seconds"] >= row["stopped_at_seconds"]]
    worst = max((float(np.max(np.abs(s["command"]))) for s in after), default=0.0)
    print(f"| {SIM_STOP_AT_SECONDS:.2f} s | {row['stopped_at_seconds']:.2f} s |"
          f" {delay:.2f} s | {delay * control_hz:.0f} |"
          f" max |command| {worst:.3f} |")
    print("  The delay is the clip's own length plus"
          f" {hearing_module.SEGMENT_END_SILENCE_SECONDS:.2f} s of silence"
          " before the segment closes plus one detector tick: the word is"
          " decided on the WHOLE utterance, so the robot cannot stop before the"
          " speaker has finished saying it.")


# ----------------------------------------------------- 5: the physics claim
def physics_parity(world, clip_path, seconds=6.0) -> dict:
    """Section 5: the ears cannot move the robot. -> worst difference per array.

    THE CLAIM, and it is `test_guide` section D's claim with a second sensor on
    top. Everything the hearing system does -- reading the head body's frame,
    synthesising four channels of audio, running a VAD, a recogniser and a
    correlator -- must leave the robot's own trajectory EXACTLY where it was.
    Nothing in `hearing.py` writes to `MjData` or to the model, and this is that
    sentence measured rather than asserted.

    THE TEST, deliberately the same shape as `test_guide.physics_parity`: the
    same scripted command, tick for tick, flown twice from the same reset --
    once with hearing off, once with it ON and a real utterance injected so the
    detectors actually fire. The command is SCRIPTED and not taken from the
    behaviour for the obvious reason: with hearing on the behaviour writes the
    command, so a run driven by it would differ for a reason that is not
    physics. What is being tested is the SENSOR, not the controller.
    """
    from app.harness import guide as guide_module

    fields = ("qpos", "qvel", "ctrl", "sensordata", "qfrc_constraint", "cfrc_ext")
    scene, episode, meta = open_world(world)
    ticks = int(seconds * episode.control_hz)

    def fly(hearing_on):
        import contextlib
        import io
        system = guide_module.GuideSystem(scene, scene.model, episode.control_hz,
                                          verbose=False)
        hearing = None
        if hearing_on:
            with contextlib.redirect_stdout(io.StringIO()):
                hearing = hearing_module.HearingSystem(
                    scene.model, episode.control_hz, seed=0, verbose=False)
                hearing.add_injector(clip_path, 1.0)
        episode.reset()
        system.place(episode.spawn_position_world)
        history = {name: [] for name in fields}
        fired = 0
        for tick in range(ticks):
            command = np.array([0.5, 0.0, 0.4 * math.sin(tick / 25.0)])
            guide_command = system.update(episode.data, tick, True, True)
            if hearing is not None:
                hearing.update(episode.data, tick, True, system, guide_command,
                               0.0, tick / episode.control_hz)
                fired += 1 if hearing.ears.new_segment else 0
            episode.step(command, np.zeros(2))
            for name in fields:
                history[name].append(
                    np.asarray(getattr(episode.data, name), dtype=float).copy())
        system.close()
        return {name: np.array(values) for name, values in history.items()}, fired

    off, _ = fly(False)
    on, fired = fly(True)
    differences = {name: float(np.max(np.abs(off[name] - on[name])))
                   for name in fields}
    differences["_utterances_heard"] = fired
    return differences


def print_physics_parity(world, differences) -> None:
    fired = differences.pop("_utterances_heard", 0)
    print(f"\n5. PHYSICS PARITY, hearing OFF vs hearing ON with a real"
          f" utterance -- {world}"
          f"  (same reset, same scripted command, tick for tick;"
          f" {fired} utterance(s) recognised during the ON run)")
    print("| array | max abs difference |")
    print("|---|---|")
    for name, difference in differences.items():
        print(f"| `{name}` | {difference:.3e} |")
    worst = max(differences.values())
    print(f"  worst over all {len(differences)} arrays: {worst:.3e}"
          + ("  -- BIT-IDENTICAL" if worst == 0.0 else "  -- NOT IDENTICAL"))


# ------------------------------------------------------------------- driver
def choose_clip(manifest, text, voice=None):
    """One named clip from the corpus. -> its absolute path."""
    for row in manifest["clips"]:
        if row["text"] == text and (voice is None or row["voice"] == voice):
            return os.path.join(manifest["directory"], row["path"])
    raise SystemExit(f"[test_hearing] no {text!r} clip in the corpus")


def main(arguments) -> None:
    manifest = corpus_module.load_manifest(arguments.corpus)
    # THE DEMO CLIPS ARE NEVER STRIDED OUT. `--clip-stride` is for making a
    # quick pass over 360 synthetic voices; dropping the five recordings the
    # demo actually plays would be exactly the wrong economy.
    demo = [row for row in manifest["clips"] if row["voice"] == "demo"]
    clips = [row for row in manifest["clips"][::max(1, arguments.clip_stride)]
             if row["voice"] != "demo"] + demo
    print(f"[test_hearing] corpus: {len(manifest['clips'])} clips,"
          f" using {len(clips)} (stride {arguments.clip_stride});"
          f" {len(manifest['voices'])} voices"
          f" {manifest['voices']}"
          f" x rates {manifest['rates_words_per_minute']}")
    print(hearing_module.describe_wind_law())
    # PRINT WHAT MUST BE INGESTED. The wind law's anchor is stated against
    # `VOICE_REFERENCE_RMS_AT_ONE_METER`; if the corpus's own level has drifted
    # away from it, every SNR in every table below is off by the difference, and
    # this is the line that says so.
    levels = [float(np.sqrt(np.mean(corpus_module.read_clip(manifest, row) ** 2)))
              for row in clips]
    peaks = [float(np.max(np.abs(corpus_module.read_clip(manifest, row))))
             for row in clips]
    print(f"[test_hearing] corpus level at unity gain: rms median"
          f" {np.median(levels):.4f} (1st/99th {np.percentile(levels, 1):.4f}"
          f"/{np.percentile(levels, 99):.4f}), peak median"
          f" {np.median(peaks):.3f}"
          f"  -- `hearing.VOICE_REFERENCE_RMS_AT_ONE_METER` is"
          f" {hearing_module.VOICE_REFERENCE_RMS_AT_ONE_METER}"
          f" ({20 * math.log10(np.median(levels) / hearing_module.VOICE_REFERENCE_RMS_AT_ONE_METER):+.1f} dB off)")
    ears = hearing_module.Ears(seed=arguments.seed, verbose=True,
                               control_hz=CONTROL_HZ)
    print(f"[test_hearing] detectors available: webrtcvad"
          f" {ears.voice_activity.available}, vosk {ears.stop_word.available}")

    sections = set(arguments.sections)
    if {1, 2} & sections:
        print("\n[test_hearing] section 1+2: sweeping the corpus over the"
              f" {len(WIND_SPEEDS_MPS)}x{len(DISTANCES_METERS)} grid ...")
        results = sweep_corpus(manifest, clips, ears, seed=arguments.seed)
        if 1 in sections:
            threshold = print_stop_tables(results)
            print_demo_clip_table(manifest, ears, threshold,
                                  seed=arguments.seed)
            print_source_level_table(
                manifest, clips[::max(1, arguments.bearing_stride)], ears,
                seed=arguments.seed)
        if 2 in sections:
            print_voice_table(results)
            print_aggressiveness_table(
                manifest, clips[::max(1, arguments.bearing_stride)],
                seed=arguments.seed)
    if 3 in sections:
        bearing_clips = clips[::max(1, arguments.bearing_stride)]
        print(f"\n[test_hearing] section 3: {len(bearing_clips)} clips x"
              f" {len(BEARING_AZIMUTHS_DEGREES)} azimuths x the grid ...")
        print_bearing_table(bearing_table(manifest, bearing_clips, ears,
                                          seed=arguments.seed))
    if 4 in sections:
        yaw_authority_table()
        # A SEED HAS TO CHANGE SOMETHING. With no wind the ear model is
        # deterministic and the physics is deterministic, so five seeds of the
        # same clip are five identical flights -- which the first run of this
        # section duly produced, two rows agreeing to the centimetre. The seed
        # therefore picks a different SPEAKER, which is the variation that
        # actually matters here: a different voice is a different utterance
        # length, a different spectrum and a different vosk confidence.
        voices = manifest["voices"]
        rows = []
        for seed in range(arguments.sim_seeds):
            voice = voices[seed % len(voices)]
            call = choose_clip(manifest, "come here", voice)
            stop = choose_clip(manifest, "stop", voice)
            print(f"[test_hearing] section 4: white-out flight, seed {seed}"
                  f" ({voice}) ...", flush=True)
            rows.append(end_to_end(arguments.world, call, stop, seed,
                                   SIM_WHITEOUT_VISIBILITY_METERS,
                                   voice=voice))
        control_voice = voices[0]
        for visibility in SIM_VISIBILITY_LADDER_METERS[1:]:
            print(f"[test_hearing] section 4: visibility {visibility:.0f} m arm"
                  f" ({control_voice}) ...", flush=True)
            rows.append(end_to_end(
                arguments.world,
                choose_clip(manifest, "come here", control_voice),
                choose_clip(manifest, "stop", control_voice), 0,
                visibility, voice=control_voice))
        # A CLOSE ARM. Six metres is the user's scenario and it is beyond this
        # walker's reach (table 4a); 2.5 m asks the same question of a plant
        # that can answer it, so the behaviour is tested and the plant limit
        # stays separately visible instead of swallowing the whole section.
        print(f"[test_hearing] section 4: close arm, 2.5 m, white-out"
              f" ({control_voice}) ...", flush=True)
        rows.append(end_to_end(
            arguments.world, choose_clip(manifest, "come here", control_voice),
            choose_clip(manifest, "stop", control_voice), 0,
            SIM_WHITEOUT_VISIBILITY_METERS, voice=control_voice,
            distance_meters=2.5))
        rows.append(end_to_end(
            arguments.world, choose_clip(manifest, "come here", control_voice),
            choose_clip(manifest, "stop", control_voice), 0,
            SIM_CLEAR_VISIBILITY_METERS, voice=control_voice,
            distance_meters=2.5))
        print_end_to_end(rows)
        print("[test_hearing] section 4b: `stop` mid-walk ...", flush=True)
        print_stop_mid_walk(end_to_end(
            arguments.world, choose_clip(manifest, "come here", control_voice),
            choose_clip(manifest, "stop", control_voice), 0,
            SIM_WHITEOUT_VISIBILITY_METERS,
            seconds=SIM_STOP_AT_SECONDS + 8.0,
            stop_at_seconds=SIM_STOP_AT_SECONDS, voice=control_voice))
    if 5 in sections:
        stop = choose_clip(manifest, "stop", arguments.sim_voice)
        print("[test_hearing] section 5: physics parity ...", flush=True)
        print_physics_parity(arguments.world,
                             physics_parity(arguments.world, stop))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--corpus", default=corpus_module.CORPUS_DIRECTORY)
    parser.add_argument("--clip-stride", type=int, default=1,
                        help="use every Nth corpus clip (tables 1 and 2)")
    parser.add_argument("--bearing-stride", type=int, default=8,
                        help="table 3 costs 5 azimuths per clip, so it takes a"
                             " further subsample on top of --clip-stride")
    parser.add_argument("--world", default=DEFAULT_SIM_WORLD,
                        help="the world sections 4 and 5 fly. Rope OFF by"
                             " default: the ear layer commands a BODY yaw, and"
                             " a palm clipped to a fixed line cannot turn.")
    parser.add_argument("--sim-voice", default="Samantha",
                        help="which corpus voice sections 4/5 inject")
    parser.add_argument("--sim-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
