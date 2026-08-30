"""Manufacture the speech corpus with macOS `say`. Ten voices, three rates.

    ../.venv_everest/bin/python -m app.harness.hearing_corpus
    ../.venv_everest/bin/python -m app.harness.hearing_corpus --voices Samantha Daniel

WHY A MANUFACTURED CORPUS. `test_hearing` has to answer "how often does the
robot hear `stop` when it is said, and how often does it hear `stop` when it is
NOT said" -- and both halves need many speakers, because a keyword spotter that
works on one voice is a keyword spotter that works on one voice. macOS ships a
speech synthesiser with a dozen English voices across four accents, so the
corpus costs nothing, regenerates identically on any Mac, and never has to be
committed.

THREE CLASSES, and the third is the one that matters:

    stop        "stop", "stop it", "stop stop"  -- MUST fire.
    other       "come here", "hey", "over here", "let's go", "help",
                "this way"                      -- must be heard as VOICE and
                                                   must NOT fire the stop.
    near-miss   "top", "shop", "drop"           -- one phoneme from the keyword.
                                                   These are what a false-stop
                                                   rate is actually made of; a
                                                   corpus of "come here" would
                                                   make any threshold look
                                                   perfect.

`"stop it"` is a STOP (the user's ruling): a person shouting "stop it" at a
robot means stop.

THE FORMAT. macOS refuses `--data-format=LEI16@16000` with an `.aiff`
container -- AIFF is big-endian by definition and the driver rejects the
mismatch with `Opening output file failed: fmt?`. `.wav` takes LEI16 happily, so
the corpus is WAV at 16 kHz mono, which is also exactly what vosk and webrtcvad
want and means nothing has to be resampled anywhere.

Inputs  : nothing but a Mac.
Outputs : `<directory>/<class>/<voice>_<rate>_<slug>.wav`, 16 kHz mono int16,
          plus `manifest.json` -- the list `test_hearing` reads, each row
          `{path, text, label, voice, rate_words_per_minute, seconds}`.
          The directory is GITIGNORED: it is regenerable in two minutes and it
          is 20 MB of audio nobody should be reviewing in a diff.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_HARNESS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIRECTORY = os.path.join(_HARNESS_DIRECTORY, "hearing_corpus")
MANIFEST_NAME = "manifest.json"

# Ten voices across US, UK, Irish, Australian, South African and Indian
# English, male and female. Every one of them is a stock macOS voice; the
# generator SKIPS any that this Mac does not have and says so, rather than
# failing, because voice availability changes with the OS version.
DEFAULT_VOICES = ("Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa",
                  "Fred", "Albert", "Ralph", "Tara")
# macOS's default is about 175 wpm. Slow, normal and hurried: a person shouting
# across a slope is not reading aloud.
DEFAULT_RATES_WORDS_PER_MINUTE = (150, 190, 240)

UTTERANCES = (
    # (text, label)
    ("stop", "stop"),
    ("stop it", "stop"),
    ("stop stop", "stop"),
    ("come here", "other"),
    ("hey", "other"),
    ("over here", "other"),
    ("let's go", "other"),
    ("help", "other"),
    ("this way", "other"),
    ("top", "near_miss"),
    ("shop", "near_miss"),
    ("drop", "near_miss"),
)
LABELS = ("stop", "other", "near_miss")
# The classes that must NOT fire a stop, pooled -- this is the denominator of
# the false-stop rate `test_hearing` prints.
NON_STOP_LABELS = ("other", "near_miss")

SAMPLE_RATE_HZ = 16000
DATA_FORMAT = f"LEI16@{SAMPLE_RATE_HZ}"

# THE DEMO CLIPS -- the five recordings the sidebar's MANUAL mode actually
# plays (`app/web/sounds/voice/`, ElevenLabs, committed to the repo). They are
# in the corpus because they are the utterances the demo will be judged on: a
# threshold tuned on `say` voices that misses the very clip the button plays is
# a threshold tuned on the wrong thing. "Pemba" is a NAME, so it is an
# ordinary call -- any human voice that is not `stop` means "come".
DEMO_VOICE_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(_HARNESS_DIRECTORY)), "app", "web",
    "sounds", "voice")
DEMO_CLIPS = (
    ("stop_1.mp3", "stop", "stop"),
    ("pemba_1.mp3", "other", "pemba"),
    ("pemba_2.mp3", "other", "pemba"),
    ("pemba_3.mp3", "other", "pemba"),
    ("pemba_4.mp3", "other", "pemba"),
)


def demo_rows() -> list:
    """The five sidebar clips as manifest rows. -> [row], possibly empty.

    They carry an ABSOLUTE path (`absolute`) because they live in the repo, not
    in the generated corpus directory, and they are 44.1 kHz mp3 rather than
    16 kHz wav -- `read_clip` resamples and mixes to mono, so nothing downstream
    has to care.
    """
    import soundfile

    rows = []
    for name, label, text in DEMO_CLIPS:
        path = os.path.join(DEMO_VOICE_DIRECTORY, name)
        if not os.path.exists(path):
            continue
        try:
            # READ, DO NOT `info`. libsndfile's frame count for an mp3 is an
            # ESTIMATE from the bitrate and it is wrong: `stop_1.mp3` reports
            # 1.83 s and decodes to 0.78 s. The length is used in tables, so it
            # has to be the length that will actually be played.
            samples, sample_rate = soundfile.read(path, dtype="float32",
                                                  always_2d=True)
        except Exception as error:      # pragma: no cover - reporting only
            print(f"[corpus] cannot read demo clip {name}: {error}", flush=True)
            continue
        rows.append({
            "path": name, "absolute": path, "text": text, "label": label,
            "voice": "demo", "rate_words_per_minute": 0,
            "seconds": float(samples.shape[0]) / float(sample_rate),
            "sample_rate_hz": int(sample_rate),
        })
    return rows


def slug(text: str) -> str:
    return "".join(character if character.isalnum() else "_"
                   for character in text).strip("_").lower()


def available_voices(requested) -> list:
    """Which of `requested` this Mac actually has. -> the kept names."""
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True,
                                 text=True, check=True).stdout
    except Exception as error:
        print(f"[corpus] `say -v ?` failed ({error}); trying every requested"
              " voice and letting the failures speak for themselves", flush=True)
        return list(requested)
    installed = {line.split()[0] for line in listing.splitlines() if line.strip()}
    kept = [voice for voice in requested if voice in installed]
    missing = [voice for voice in requested if voice not in installed]
    if missing:
        print(f"[corpus] not installed on this Mac, skipped: {missing}",
              flush=True)
    return kept


def generate(directory=CORPUS_DIRECTORY, voices=DEFAULT_VOICES,
             rates=DEFAULT_RATES_WORDS_PER_MINUTE, force=False) -> dict:
    """Say every utterance in every voice at every rate. -> the manifest dict."""
    import soundfile

    voices = available_voices(voices)
    if not voices:
        raise SystemExit("[corpus] no usable voices; nothing to generate")
    for label in LABELS:
        os.makedirs(os.path.join(directory, label), exist_ok=True)

    rows, made, reused, failed = [], 0, 0, 0
    for voice in voices:
        for rate in rates:
            for text, label in UTTERANCES:
                name = f"{voice}_{rate}_{slug(text)}.wav"
                path = os.path.join(directory, label, name)
                if force or not os.path.exists(path):
                    result = subprocess.run(
                        ["say", "-v", voice, "-r", str(rate), "-o", path,
                         "--data-format=" + DATA_FORMAT, text],
                        capture_output=True, text=True)
                    if result.returncode != 0 or not os.path.exists(path):
                        print(f"[corpus] FAILED {voice} {rate} {text!r}:"
                              f" {result.stderr.strip()}", flush=True)
                        failed += 1
                        continue
                    made += 1
                else:
                    reused += 1
                info = soundfile.info(path)
                rows.append({
                    "path": os.path.relpath(path, directory),
                    "text": text,
                    "label": label,
                    "voice": voice,
                    "rate_words_per_minute": int(rate),
                    "seconds": float(info.frames) / float(info.samplerate),
                    "sample_rate_hz": int(info.samplerate),
                })

    rows.extend(demo_rows())
    manifest = {
        "directory": directory,
        "voices": voices,
        "rates_words_per_minute": list(rates),
        "utterances": [{"text": text, "label": label}
                       for text, label in UTTERANCES],
        "clips": rows,
    }
    with open(os.path.join(directory, MANIFEST_NAME), "w") as handle:
        json.dump(manifest, handle, indent=1)

    # PRINT WHAT MUST BE INGESTED. A corpus whose clips are all 0.2 s, or all at
    # one sample rate that is not 16 kHz, or 40% missing, produces a beautiful
    # table of nonsense -- so the distribution goes to stdout before anything
    # reads it.
    print(f"\n[corpus] {len(rows)} clips in {directory}"
          f"  ({made} generated, {reused} reused, {failed} failed)")
    print(f"[corpus] plus {len(demo_rows())} DEMO clips from"
          f" {DEMO_VOICE_DIRECTORY} (the sidebar's MANUAL buttons)")
    print(f"[corpus] {len(voices)} voices {voices}"
          f"  x {len(rates)} rates {list(rates)}"
          f"  x {len(UTTERANCES)} utterances")
    print("| class | clips | seconds min/median/max | sample rates |")
    print("|---|---|---|---|")
    for label in LABELS:
        lengths = sorted(row["seconds"] for row in rows if row["label"] == label)
        rates_seen = sorted({row["sample_rate_hz"] for row in rows
                             if row["label"] == label})
        if not lengths:
            print(f"| {label} | 0 | | |")
            continue
        print(f"| {label} | {len(lengths)} |"
              f" {lengths[0]:.2f} / {lengths[len(lengths) // 2]:.2f} /"
              f" {lengths[-1]:.2f} | {rates_seen} |")
    return manifest


def load_manifest(directory=CORPUS_DIRECTORY) -> dict:
    """Read the manifest, or explain how to make it. -> the manifest dict."""
    path = os.path.join(directory, MANIFEST_NAME)
    if not os.path.exists(path):
        raise SystemExit(
            f"[corpus] no manifest at {path}. Generate it first:\n"
            "    ../.venv_everest/bin/python -m app.harness.hearing_corpus")
    with open(path) as handle:
        manifest = json.load(handle)
    manifest["directory"] = directory
    # The demo clips live in the repo, not in the corpus directory, so a
    # manifest written before they existed still picks them up.
    known = {row["path"] for row in manifest["clips"]}
    manifest["clips"].extend(row for row in demo_rows()
                             if row["path"] not in known)
    return manifest


def read_clip(manifest, row):
    """One clip as mono float32 at 16 kHz. -> (samples,)."""
    import numpy as np
    import soundfile

    samples, sample_rate = soundfile.read(
        row.get("absolute") or os.path.join(manifest["directory"], row["path"]),
        dtype="float32", always_2d=True)
    mono = samples.mean(axis=1).astype(np.float32)
    if int(sample_rate) != SAMPLE_RATE_HZ:
        count = int(round(mono.size * SAMPLE_RATE_HZ / float(sample_rate)))
        mono = np.interp(np.linspace(0.0, mono.size - 1.0, count),
                         np.arange(mono.size), mono).astype(np.float32)
    return mono


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", default=CORPUS_DIRECTORY)
    parser.add_argument("--voices", nargs="+", default=list(DEFAULT_VOICES))
    parser.add_argument("--rates", nargs="+", type=int,
                        default=list(DEFAULT_RATES_WORDS_PER_MINUTE))
    parser.add_argument("--force", action="store_true",
                        help="regenerate clips that already exist")
    arguments = parser.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("[corpus] `say` is macOS only; this corpus cannot be"
                         " generated here")
    generate(arguments.directory, arguments.voices, arguments.rates,
             arguments.force)
