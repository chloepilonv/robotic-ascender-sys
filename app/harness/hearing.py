"""The robot's EARS: four virtual microphones, three detectors, one behaviour.

THE FEATURE. A person shouts at the robot. The robot works out (a) that a human
voice is there, (b) whether the word was "stop", and (c) which way the sound
came from -- and then it COMES TO HER. If its eyes can see her, the stereo
follower already knows how to walk to a person, so vision drives. If they
cannot, the ear bearing drives until they can.

THE HONESTY LINE, and it is exactly the one `guide.StereoEyes` draws. The eyes
take the simulator's TRUTH geometry, render it into two pictures, and then
measure a distance from the PICTURES. The ears do the same trick with sound:

    the MICROPHONE signal is real           -- it is the human's actual voice,
                                               captured by the Mac's microphone
                                               in the browser and streamed here
                                               as 16 kHz mono PCM. Nothing
                                               about the WORDS is simulated.
    the EAR signals are SYNTHESISED from truth positions. The voice is emitted
                                               at the hiker's mouth (a point on
                                               the guide's head, which the
                                               simulator knows) and propagated
                                               to four points on the robot's
                                               head (which the simulator also
                                               knows): 1/r gain, a propagation
                                               delay, a gentle air-absorption
                                               low-pass, and wind noise.
    every DECISION reads only the four ear signals. Nothing downstream of
                                               `EarArray.feed` ever touches
                                               `MjData` again. The bearing is
                                               recovered by cross-correlating
                                               the four channels, exactly as a
                                               real mic array would, and it is
                                               wrong in the ways a real mic
                                               array is wrong.

The truth positions are the sensor MODEL's input, the way the renderer's scene
graph is the eyes' input. What is a LABELLED CHEAT and what is not is therefore
the same distinction the eyes make, and `PARITY.md` carries the ledger.

FIVE PIECES, kept apart so each can be replaced on its own:

    EarArray        truth geometry + one mono source -> four ear channels.
                    Propagation only: delays, gains, one low-pass per mic.
    WindNoise       the noise floor, as a function of the wind dial. Drawn
                    INDEPENDENTLY per mic, because a mic array's whole ability
                    to hear a direction rests on its channels being different,
                    and correlated noise would leave the bearing untouched by
                    a gale, which is a lie.
    VoiceActivity   webrtcvad on 30 ms frames -> "a human voice is present",
                    reported as the voiced-frame share of a rolling window.
    StopWord        Vosk small English with a grammar of exactly
                    `["stop", "[unk]"]` -> a confidence for the word `stop`.
    Bearing         GCC-PHAT between the front/back and the left/right pairs
                    -> an azimuth in the ROBOT'S BODY frame, and a confidence
                    from the correlation peak's sharpness.

Then `HearingBehaviour`, which is the layer on top of `guide.GuideFollower`,
and `HearingSystem`, which is what `runtime.run` holds.

WHY webrtcvad AND NOT SILERO. Silero VAD is a torch model, and this venv has no
torch (it is a JAX/brax environment). Pulling ~250 MB of torch into a 50 Hz
control loop to answer a yes/no question that a 158 kB C library answers in
40 microseconds is the wrong trade for a demo that must run on a laptop next to
MuJoCo. webrtcvad is Google's WebRTC voice-activity detector, it is what a
telephony stack actually ships, and its per-frame verdict is binary -- so the
"probability" this module reports is the VOICED-FRAME SHARE of the rolling
window, which is an honest number rather than a made-up confidence.

Inputs  : `MjData` each tick (truth positions only), and a mono int16 PCM
          stream at `SAMPLE_RATE_HZ` from the page's microphone.
Outputs : `HearingSystem.state()` (the websocket block), `.recorded()` (the
          episode columns), `.command(...)` (the (3,) walking command), and
          `.take_ear_pcm()` (the `EAR0` monitor mix).
"""
from __future__ import annotations

import math
import time

import numpy as np

# ------------------------------------------------------------------ the wire
SAMPLE_RATE_HZ = 16000
# 4 ASCII bytes then int16 little-endian PCM, both ways.
MIC_MESSAGE_PREFIX = b"MIC0"      # page -> runtime, the human's voice
EAR_MESSAGE_PREFIX = b"EAR0"      # runtime -> page, what the robot hears
# The page's ScriptProcessor block is 512 samples (32 ms) -- the smallest
# power-of-two block that does not glitch on a busy tab. The runtime consumes
# exactly one control tick's worth per tick and does not care what size the
# chunks arrived in.
MICROPHONE_QUEUE_SECONDS = 2.0    # ring buffer; a page that runs ahead is capped
# HOW MUCH AUDIO MUST BE IN THE RING BEFORE IT IS DRAINED. The page and the
# simulation both run at nominally 16000 samples a second and neither runs at
# exactly that, so a ring drained the instant anything arrives punches SILENT
# HOLES into the middle of words -- and a `stop` with a hole in it is not a
# `stop`. MEASURED through the browser before this existed: the clip that
# decodes at confidence 1.000 offline came back as an ordinary `voice`. Priming
# converts jitter into a fixed 150 ms of latency, which nobody can hear and the
# behaviour does not care about.
MICROPHONE_PRIME_SECONDS = 0.15

# --------------------------------------------------------------- the array
# FOUR MICS ON THE HEAD, in the TORSO body's own frame: +x forward, +y left,
# +z up. The G1 carries a mic array; four at 7 cm is a plausible one, and four
# rather than two is what removes the front/back ambiguity a single pair has.
MICROPHONE_RADIUS_METERS = 0.07
MICROPHONE_LAYOUT = (
    ("front", (+1.0, 0.0, 0.0)),
    ("back",  (-1.0, 0.0, 0.0)),
    ("left",  (0.0, +1.0, 0.0)),
    ("right", (0.0, -1.0, 0.0)),
)
MICROPHONE_NAMES = tuple(name for name, _ in MICROPHONE_LAYOUT)
MONITOR_MICROPHONE = "front"      # which channel the EAR0 mix carries
SPEED_OF_SOUND_METERS_PER_SECOND = 330.0
# The body the array rides. Same body as the eyes (`guide._add_eye_cameras`
# mounts them on the `d435i` parent, which is `torso_link`), so the ears and the
# eyes share a frame and a bearing means the same thing to both.
HEAD_BODY_NAMES = ("torso_link", "torso", "pelvis")

# THE VOICE'S LOUDNESS LAW. Amplitude falls as 1/r -- the free-field spherical
# spreading law, 6 dB per doubling. `VOICE_AMPLITUDE_AT_ONE_METER = 1.0` means
# "the microphone's own recorded level IS the level at one metre", so a clip
# that peaks at 0.7 full scale is a person who peaks at 0.7 full scale a metre
# from your ear. Everything about the SNR grid is anchored on that sentence.
VOICE_AMPLITUDE_AT_ONE_METER = 1.0
MINIMUM_RANGE_METERS = 0.25       # no 1/r singularity when she leans in
# AIR ABSORPTION. A one-pole low-pass whose cutoff falls with range. Over the
# 2-10 m this demo lives in the effect is genuinely small -- 7.6 kHz at 2 m,
# 6.2 kHz at 10 m -- and it is here for completeness rather than because it
# changes a detection. Do not expect it to show up in a table; it does not.
AIR_ABSORPTION_CUTOFF_AT_ZERO_HZ = 8000.0
AIR_ABSORPTION_RANGE_SCALE_METERS = 40.0

# ------------------------------------------------------------- the wind noise
# THE NOISE-VS-WIND LAW, stated here rather than buried, because every number in
# `test_hearing`'s tables is a function of it:
#
#     noise_amplitude(speed) = WIND_NOISE_FLOOR
#                            + WIND_NOISE_AT_REFERENCE
#                              * (speed / WIND_NOISE_REFERENCE_MPS) ** 2
#
# QUADRATIC, because wind noise on a bare capsule is turbulent pressure on the
# diaphragm and turbulent pressure goes as the dynamic head, rho*U^2/2. It is
# not the aerodynamic drag law of `wind_env.py` reused by coincidence; it is the
# same physics arriving at the same exponent.
#
# THE ANCHOR, and it is the one number in this file that is a judgement call: a
# 20 m/s gale on an UNWINDSHIELDED microphone sits about 10 dB below a VOICE AT
# ONE METRE. A foam windshield would be worth another 15-20 dB and this robot is
# not wearing one.
#
# BOTH SIDES OF THAT COMPARISON ARE RMS, and getting that wrong is how this file
# was first written: the anchor was applied to the corpus's PEAK level instead
# of its rms, which is 15.6 dB higher, and the whole grid collapsed -- a 6 m/s
# breeze wiped out a shout at two metres and every table below 0 m/s read 0.0%.
# So the voice's reference level is stated explicitly and MEASURED:
# `VOICE_REFERENCE_RMS_AT_ONE_METER` is the median rms of a `say` utterance at
# unity gain, and `test_hearing` prints the corpus's own measured median beside
# it so a drift shows up rather than hiding.
#
#     WIND_NOISE_AT_REFERENCE = VOICE_REFERENCE_RMS_AT_ONE_METER * 10^(-10/20)
#
#     0 m/s -> 0.0005     6 m/s -> 0.0038     12 m/s -> 0.0136    20 m/s -> 0.0369
#
# Equivalently: the gale equals the voice at 3.2 m, and would equal it at 1 m in
# a 63 m/s hurricane. The floor is the electronics, not the weather: a real
# preamp is never silent.
WIND_NOISE_FLOOR = 0.0005
VOICE_REFERENCE_RMS_AT_ONE_METER = 0.115
WIND_NOISE_AT_REFERENCE = VOICE_REFERENCE_RMS_AT_ONE_METER * 10.0 ** (-10.0 / 20.0)
WIND_NOISE_REFERENCE_MPS = 20.0
# THE CHARACTER, ported from the page's own wind synthesis (`render3d.html`,
# `makePinkNoiseBuffer` + the 300 Hz bandpass and 700 Hz low-pass): pink noise,
# rolled off on top, with a slow level LFO so it breathes instead of hissing.
# The page's numbers are reused verbatim where they transfer.
WIND_NOISE_LOW_PASS_HZ = 700.0
WIND_NOISE_GUST_HZ = 0.13         # the page's `lfo.frequency.value`
WIND_NOISE_GUST_DEPTH = 0.45      # +/- share of the mean level
# WIND NOISE IS DRAWN INDEPENDENTLY PER MICROPHONE. Turbulence at the capsule is
# a local pressure fluctuation, not a sound field arriving from somewhere, so it
# does not correlate across 14 cm -- and that INDEPENDENCE is precisely what
# makes a gale destroy the bearing rather than merely making it quieter. Drawing
# one noise vector and copying it to four mics would leave GCC-PHAT's peak
# untouched at any wind speed, and the bearing table would be a straight line
# of lies.

# --------------------------------------------------------------- the detectors
DETECTOR_EVERY_N_TICKS = 5        # 10 Hz against the 50 Hz control tick
VOICE_ACTIVITY_AGGRESSIVENESS = 2  # webrtcvad 0 (permissive) .. 3 (strict)
VOICE_ACTIVITY_FRAME_MILLISECONDS = 30    # webrtcvad accepts 10, 20 or 30
VOICE_ACTIVITY_WINDOW_SECONDS = 0.48      # 16 frames -> the REPORTED share
# WHICH SHARE OPENS AND CLOSES A SEGMENT, and it is deliberately NOT the
# rolling window's. The window is a display number: it lags speech by up to
# 480 ms in both directions, so a segment machine driven by it stays "voiced"
# for half a second after the speaker has stopped, and the utterance takes
# 0.7 s to close instead of 0.2. MEASURED: with the window driving it, a 0.5 s
# tail of silence was not enough to close a single segment in the whole corpus
# and every table came back empty. The segment machine therefore reads the
# share of the frames consumed by THIS detector tick -- three 30 ms frames, so
# one voiced frame out of three opens an utterance and none closes it.
VOICE_PRESENT_SHARE = 0.30
# A SEGMENT is one utterance: it opens when the VAD says voice and closes a
# short silence later. Both the word decision and the bearing are made ONCE per
# segment, on the whole utterance, which is why they are as good as they are --
# a 100 ms slice of "stop" is not a word and not a direction.
SEGMENT_END_SILENCE_SECONDS = 0.20
SEGMENT_MAXIMUM_SECONDS = 2.0
SEGMENT_MINIMUM_SECONDS = 0.12    # shorter than this is a click, not speech
# PRE-ROLL: how much audio from BEFORE the VAD said "voice" is glued onto the
# front of an utterance. It is not a nicety. The VAD runs at 10 Hz, so it can
# be up to 100 ms late, and the recogniser is then handed a word with its first
# consonant missing -- for `stop` that is the /s/, and "top" is exactly one of
# the near-misses the grammar is built to reject. MEASURED: with no pre-roll a
# "stop" spoken at a robot 6 m away came back `[unk]` with 0.000 confidence,
# while the SAME clip through the SAME ear model with 300 ms of silence in
# front of it came back 1.000. The offline tables were passing only because
# their clips were padded; the live path had no pad and no chance.
SEGMENT_PREROLL_SECONDS = 0.30
# The bearing is measured on the LOUDEST window of the segment, because the
# quiet head and tail of an utterance are mostly the noise floor and GCC-PHAT
# would be cross-correlating wind with wind.
BEARING_WINDOW_SECONDS = 0.30
BEARING_INTERPOLATION_FACTOR = 8  # sub-sample resolution: 1/8 of 62.5 us
# How far below the strongest spectral bin the phase transform stops whitening.
# See `Bearing._pair` for the measurement that set it.
PHASE_TRANSFORM_FLOOR = 0.02
# How far past the physically possible lag range the peak's SHARPNESS is
# measured against. See `Bearing._pair` for the measurement that set it.
BEARING_FLOOR_SPAN_MULTIPLIER = 12
# Peak sharpness -> a [0, 1] confidence. A correlation peak no taller than
# `_FLOOR` times the background is not a direction; `_SPAN` above that is a
# clean one. Both read off the measured sharpnesses in `test_hearing` table 3b.
BEARING_SHARPNESS_FLOOR = 2.0
BEARING_SHARPNESS_SPAN = 8.0
# The vosk grammar. Exactly one word plus the escape hatch, which is what makes
# a 40 MB model usable at all: with the grammar the decoder can only ever emit
# `stop` or `[unk]`, so a shout of "come here" cannot be scored as a partial
# match to anything.
STOP_WORD = "stop"
VOSK_GRAMMAR = '["stop", "[unk]"]'
# CHOSEN FROM `test_hearing` TABLE 1, not guessed. The number below is the
# threshold the test prints and the runtime uses; re-run the test if the ear
# model, the corpus or the wind law changes, and move this line to whatever the
# table says.
STOP_CONFIDENCE_THRESHOLD = 0.90

# ------------------------------------------------------------- the behaviour
HEARING_MODES = ("IDLE", "LISTENING", "COMING_BY_EYES", "COMING_BY_EARS",
                 "STOPPED", "WAIT")
HEARING_MODE_CODES = {name: float(index)
                      for index, name in enumerate(HEARING_MODES)}
HEARD_CODES = {"none": 0.0, "voice": 1.0, "stop": 2.0}
# The walk toward a voice. Same forward speed the vision follower uses, so the
# hand-over from ears to eyes is not also a change of pace.
EAR_WALK_SPEED_METERS_PER_SECOND = 0.5
EAR_BEARING_GAIN_PER_RADIAN = 2.0
# HALF THE POLICY'S TRAINING RANGE. `ang_vel_yaw` was trained over [-1, 1], and
# on `flat_free` -- the world section 4 flies -- 0.5 is enough to steer the robot
# to a person six metres away in about forty seconds. It is capped there rather
# than at 1.0 because the response is not monotone in the command and the top of
# the range buys nothing: MEASURED on `sandbox_free`, `lin_vel_x` 0.5 with
# `ang_vel_yaw` at +0.2, +0.35 and +0.5 all tip the robot over inside four
# seconds while +1.0 survives, and none of the six rates in `test_hearing`
# table 4a produces a turn that tracks what was asked. On rough ground nothing
# in this range works, which is that table's point and the standing ASK.
EAR_MAXIMUM_YAW_RATE_RADIANS_PER_SECOND = 0.5
# THE WAIST IS THE EAR LAYER'S REAL AIMING ACTUATOR, for exactly the reason
# `guide.WaistYaw` exists: the G1 has no neck and this policy has no usable
# yaw. While the ear cue is fresher than this, `COMING_BY_EARS` points the
# waist -- and therefore the stereo pair -- at the direction the shout came
# from, which is the whole job of the ear cue: get her into the picture so
# vision can take over. Once the cue is older than this the waist is handed
# back to wherever it was, because a stale bearing is not worth holding a torso
# twisted for.
EAR_WAIST_AIM_SECONDS = 3.0
# HOW FAR THE EAR LAYER MAY TWIST THE WAIST, and it is much less than the
# waist's own 60 degree limit. `guide.WAIST_LIMIT_RADIANS` was measured on a
# robot STANDING STILL and sweeping; this one is walking and turning at the same
# time, and the two loads add. MEASURED on `terrain_free_0`, ear-driven approach,
# 90 s budget:
#
#     60 deg -> FELL at 4.5 s | 25 deg -> fell at 89.8 s | 15 deg -> survived
#      0 deg -> survived
#
# 20 degrees is inside the surviving band and still worth about a third of the
# camera's half-FOV, which is the point of aiming at all.
EAR_WAIST_AIM_LIMIT_RADIANS = math.radians(20.0)
EAR_BEARING_DEADBAND_RADIANS = math.radians(4.0)
# A bearing this uncertain is not a direction. Below it no cue is taken and the
# robot listens rather than walking toward noise.
EAR_BEARING_MINIMUM_CONFIDENCE = 0.25
# HOW LONG AN EAR CUE LASTS, and it is deliberately long. "Voice is a trigger,
# not a leash" (user): one shout starts the walk and SILENCE DOES NOT STOP IT --
# the walk ends at the person, at a `stop`, or when the eyes have taken over.
# The first version expired the cue after twenty seconds, which on this
# walker's measured ground speed (about 0.1 m/s of PROGRESS toward a heading,
# `test_hearing` table 4a) is two metres: the robot gave up two thirds of the
# way to a person who had done nothing wrong. This number is therefore a
# runaway guard, not a policy -- long enough to be irrelevant to any real
# approach, short enough that a cue cannot drive a forgotten robot for ever.
EAR_CUE_VALID_SECONDS = 120.0
# How far back the yaw history reaches. An utterance plus its measurement
# window is under two seconds; three is headroom.
YAW_HISTORY_SECONDS = 3.0
NO_MEASUREMENT = -1.0             # hud.json cannot carry NaN; see guide.py


_LFILTER = None


def _lfilter():
    """`scipy.signal.lfilter`, looked up once. Imported lazily so that importing
    this module on a machine with no scipy still lets `--help` run."""
    global _LFILTER
    if _LFILTER is None:
        from scipy.signal import lfilter
        _LFILTER = lfilter
    return _LFILTER


def wind_noise_amplitude(wind_speed_meters_per_second: float) -> float:
    """The noise floor at a given wind speed. -> amplitude, full scale = 1.0.

    The law is in this module's header and printed by `describe_wind_law()`.
    """
    speed = max(0.0, float(wind_speed_meters_per_second))
    return float(WIND_NOISE_FLOOR + WIND_NOISE_AT_REFERENCE
                 * (speed / WIND_NOISE_REFERENCE_MPS) ** 2)


def describe_wind_law() -> str:
    """One line, printed by every entry point that hears anything."""
    points = ", ".join(
        f"{speed:.0f} m/s -> {wind_noise_amplitude(speed):.4f} rms"
        f" ({decibels(wind_noise_amplitude(speed) / VOICE_REFERENCE_RMS_AT_ONE_METER):+.0f} dB"
        f" vs a voice at 1 m)" for speed in (0, 6, 12, 20))
    return (f"[hearing] wind noise (rms) = {WIND_NOISE_FLOOR}"
            f" + {WIND_NOISE_AT_REFERENCE:.4f} * (speed /"
            f" {WIND_NOISE_REFERENCE_MPS:.0f})^2  -- quadratic, because"
            f" turbulent pressure goes as the dynamic head; anchored at a"
            f" 20 m/s gale 10 dB under a voice at 1 m, whose rms is taken as"
            f" {VOICE_REFERENCE_RMS_AT_ONE_METER}. {points}."
            f"  Drawn INDEPENDENTLY per microphone.")


def _wrap_to_pi(angle_radians: float) -> float:
    """Fold an angle into (-pi, pi]. The same helper `runtime.py` carries."""
    return float((float(angle_radians) + math.pi) % (2.0 * math.pi) - math.pi)


def decibels(amplitude) -> float:
    """Full-scale dB, floored at -80 so a silent channel is a number."""
    value = float(amplitude)
    if not np.isfinite(value) or value <= 1e-4:
        return -80.0
    return float(max(-80.0, 20.0 * math.log10(value)))


# ---------------------------------------------------------- the microphone in
class MicrophoneStream:
    """The page's PCM, buffered, and handed out one control tick at a time.

    A ring of `MICROPHONE_QUEUE_SECONDS`. The browser pushes 32 ms blocks
    whenever it likes; the control loop pulls exactly one tick's worth every
    tick and gets SILENCE when the buffer has run dry, which is what a real
    microphone with no one talking into it also produces. A page that runs ahead
    of a slow simulation is capped rather than allowed to grow a delay: the
    oldest samples are dropped, because a demo wants the person's voice NOW, not
    the backlog.

    Thread safety: `push` is called from the websocket thread and `take` from
    the simulation thread, so both hold one lock. The work under it is a memcpy.
    """

    def __init__(self, capacity_seconds=MICROPHONE_QUEUE_SECONDS):
        import threading
        self.capacity = int(capacity_seconds * SAMPLE_RATE_HZ)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.samples_received = 0
        self.samples_dropped = 0
        self.samples_starved = 0
        # WHAT ARRIVED, UNTOUCHED. The runtime prints these once a second and
        # the page draws the meter from `level_db`, because the whole point of
        # turning AGC off in the browser is that a SHOUT must arrive as a shout:
        # nothing between the capsule and the ear model may normalise it.
        self.primed = False
        self.recent_rms = 0.0
        self.recent_peak = 0.0
        # (sample_count, sum_of_squares, peak) per push, trimmed to one second,
        # so the printed level is "the last second" and not "the last 32 ms".
        self._level_window = []

    def push_pcm_bytes(self, payload: bytes) -> int:
        """int16 little-endian PCM (with or without the `MIC0` prefix). -> count."""
        if payload[:4] == MIC_MESSAGE_PREFIX:
            payload = payload[4:]
        if len(payload) < 2:
            return 0
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        return self.push(samples)

    def push(self, samples) -> int:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size:
            self._level_window.append((int(samples.size),
                                       float(np.sum(samples.astype(np.float64) ** 2)),
                                       float(np.max(np.abs(samples)))))
            total = sum(row[0] for row in self._level_window)
            while len(self._level_window) > 1 and total > SAMPLE_RATE_HZ:
                total -= self._level_window.pop(0)[0]
            self.recent_rms = float(np.sqrt(
                sum(row[1] for row in self._level_window) / max(total, 1)))
            self.recent_peak = max(row[2] for row in self._level_window)
        with self.lock:
            self.buffer = np.concatenate((self.buffer, samples))
            self.samples_received += int(samples.size)
            excess = self.buffer.size - self.capacity
            if excess > 0:
                self.buffer = self.buffer[excess:]
                self.samples_dropped += int(excess)
        return int(samples.size)

    def take(self, count: int) -> np.ndarray:
        """The next `count` samples, zero-padded if the page has not kept up.

        PRIMED, not eager: see `MICROPHONE_PRIME_SECONDS`. Until the ring holds
        a prime's worth this returns silence and takes nothing, and once it
        empties it goes back to waiting -- so a gap lands BETWEEN utterances,
        where it is silence anyway, instead of inside one.
        """
        count = int(count)
        prime = int(MICROPHONE_PRIME_SECONDS * SAMPLE_RATE_HZ)
        with self.lock:
            if not self.primed:
                if self.buffer.size < prime:
                    return np.zeros(count, dtype=np.float32)
                self.primed = True
            available = min(count, self.buffer.size)
            chunk = self.buffer[:available].copy()
            self.buffer = self.buffer[available:]
            if self.buffer.size == 0:
                self.primed = False
        if available < count:
            self.samples_starved += count - available
            chunk = np.concatenate((chunk, np.zeros(count - available,
                                                    dtype=np.float32)))
        return chunk

    def clear(self) -> None:
        with self.lock:
            self.buffer = np.zeros(0, dtype=np.float32)
            self.primed = False
        self._level_window = []
        self.recent_rms = 0.0
        self.recent_peak = 0.0

    def describe(self) -> str:
        """One line of what the microphone is delivering, RAW.

        NOTHING IN THIS PIPELINE NORMALISES THE INCOMING PCM (user's ruling,
        2026-08-30) -- the browser is asked for `autoGainControl: false` and the
        ear model applies only the world: 1/r, the propagation delay, the air's
        low-pass and the wind. So a shout carries further than a mumble, which
        is the whole reason the level is printed rather than assumed.
        """
        return (f"[hearing] microphone in (last 1 s): rms {self.recent_rms:.4f}"
                f" ({decibels(self.recent_rms):+.1f} dBFS), peak"
                f" {self.recent_peak:.3f}"
                f" ({decibels(self.recent_peak):+.1f} dBFS);"
                f" {self.samples_received} samples received,"
                f" {self.samples_dropped} dropped (page ran ahead),"
                f" {self.samples_starved} silent (page ran behind)")


class VoiceInjector:
    """A wav file played into the microphone stream at a scheduled sim time.

    THE DEMO WITHOUT A MICROPHONE. `--inject-voice clip.wav@3.5` schedules a
    corpus clip to start at t = 3.5 s of simulated time, so the whole behaviour
    can be driven headlessly, screenshotted, and replayed at the same seed. It
    pushes into exactly the same ring the browser pushes into, so nothing
    downstream can tell the difference -- which is the point: a debug path that
    bypasses the real path proves nothing about the real path.
    """

    def __init__(self, path: str, start_seconds: float = 0.5):
        import soundfile

        samples, sample_rate = soundfile.read(path, dtype="float32",
                                              always_2d=True)
        self.path = path
        self.start_seconds = float(start_seconds)
        mono = samples.mean(axis=1).astype(np.float32)
        if int(sample_rate) != SAMPLE_RATE_HZ:
            # Linear resample. The corpus is generated at 16 kHz by
            # `hearing_corpus.py`, so this is a courtesy for a hand-supplied
            # file, not a path the tables ever take.
            count = int(round(mono.size * SAMPLE_RATE_HZ / float(sample_rate)))
            mono = np.interp(np.linspace(0.0, mono.size - 1.0, count),
                             np.arange(mono.size), mono).astype(np.float32)
        self.samples = mono
        self.cursor = 0

    @property
    def finished(self) -> bool:
        return self.cursor >= self.samples.size

    def step(self, time_seconds: float, count: int, stream: MicrophoneStream) -> int:
        """Push this tick's slice if the clip has started. -> samples pushed."""
        if self.finished or time_seconds < self.start_seconds:
            return 0
        chunk = self.samples[self.cursor:self.cursor + count]
        self.cursor += chunk.size
        return stream.push(chunk)


# --------------------------------------------------------------- propagation
class EarArray:
    """Truth geometry + one mono source -> four ear channels. Propagation only.

    Per microphone, per sample: the source as it was `range / c` seconds ago,
    scaled by `1/range`, low-passed by the air, plus that microphone's own
    independent wind noise. The delay and the gain are LINEARLY INTERPOLATED
    across each chunk from the previous chunk's values, so a moving robot gets a
    continuously varying delay rather than a step every 20 ms -- a step in a
    delay line is an audible click and, worse, a fake transient for the
    correlator to lock onto.

    THE INTER-MICROPHONE DELAYS ARE THE ENTIRE BEARING CUE. At 16 kHz one sample
    is 62.5 us, which is 2.06 cm of travel, so the 14 cm front-to-back baseline
    is only 6.8 samples end to end. That is why the delays here are FRACTIONAL
    (linear interpolation into the history) and why `Bearing` interpolates its
    correlation peak: rounding either one to whole samples would quantise the
    azimuth to about 15 degrees and the bearing table would be measuring the
    arithmetic instead of the array.

    Inputs  : `source` (n,) float mono in [-1, 1]; `ranges_meters` (4,) the
              range from the mouth to each microphone at the END of this chunk.
    Outputs : `(4, n)` float32, one row per microphone in `MICROPHONE_NAMES`
              order.
    """

    HISTORY_SAMPLES = 4096        # 256 ms: 84 m of propagation, ample

    def __init__(self, wind_noise=None):
        self.history = np.zeros(self.HISTORY_SAMPLES, dtype=np.float32)
        self.previous_delay_samples = np.zeros(len(MICROPHONE_LAYOUT))
        self.previous_gain = np.zeros(len(MICROPHONE_LAYOUT))
        self.previous_cutoff_hz = np.full(len(MICROPHONE_LAYOUT),
                                          AIR_ABSORPTION_CUTOFF_AT_ZERO_HZ)
        self.low_pass_state = np.zeros(len(MICROPHONE_LAYOUT))
        self.wind_noise = wind_noise
        self.seeded = False

    def reset(self) -> None:
        self.history[:] = 0.0
        self.previous_delay_samples[:] = 0.0
        self.previous_gain[:] = 0.0
        self.low_pass_state[:] = 0.0
        self.seeded = False

    def feed(self, source, ranges_meters, wind_speed_meters_per_second=0.0):
        """One control tick of sound. -> (4, n) float32."""
        source = np.asarray(source, dtype=np.float32)
        count = source.size
        ranges = np.asarray(ranges_meters, dtype=float)
        ranges = np.maximum(ranges, MINIMUM_RANGE_METERS)

        if count:
            self.history = np.concatenate((self.history[count:], source))
        chunk_start = self.history.size - count      # index of source[0]

        delay_samples = ranges / SPEED_OF_SOUND_METERS_PER_SECOND * SAMPLE_RATE_HZ
        gain = VOICE_AMPLITUDE_AT_ONE_METER / ranges
        cutoff_hz = AIR_ABSORPTION_CUTOFF_AT_ZERO_HZ * np.exp(
            -ranges / AIR_ABSORPTION_RANGE_SCALE_METERS)
        if not self.seeded:
            # The first chunk has no previous geometry to ramp from; ramping
            # from zero would put a fake 20 ms glide on the very first word.
            self.previous_delay_samples[:] = delay_samples
            self.previous_gain[:] = gain
            self.previous_cutoff_hz[:] = cutoff_hz
            self.seeded = True

        channels = np.zeros((len(MICROPHONE_LAYOUT), count), dtype=np.float32)
        if count:
            history_index = np.arange(self.history.size, dtype=np.float64)
            offsets = np.arange(count, dtype=np.float64)
            for index in range(len(MICROPHONE_LAYOUT)):
                ramp = (offsets + 1.0) / count
                delay = (self.previous_delay_samples[index]
                         + ramp * (delay_samples[index]
                                   - self.previous_delay_samples[index]))
                amplitude = (self.previous_gain[index]
                             + ramp * (gain[index] - self.previous_gain[index]))
                positions = chunk_start + offsets - delay
                channels[index] = (np.interp(positions, history_index,
                                             self.history) * amplitude)
                channels[index] = self._low_pass(
                    channels[index], index,
                    0.5 * (self.previous_cutoff_hz[index] + cutoff_hz[index]))
        self.previous_delay_samples[:] = delay_samples
        self.previous_gain[:] = gain
        self.previous_cutoff_hz[:] = cutoff_hz

        if self.wind_noise is not None and count:
            channels += self.wind_noise.draw(
                len(MICROPHONE_LAYOUT), count, wind_speed_meters_per_second)
        return channels

    def _low_pass(self, signal, index, cutoff_hz):
        """One-pole air absorption, state carried across chunks."""
        alpha = float(np.clip(
            1.0 - math.exp(-2.0 * math.pi * max(cutoff_hz, 1.0) / SAMPLE_RATE_HZ),
            1e-4, 1.0))
        # `lfilter` rather than a Python loop: 320 samples x 4 mics x 50 Hz is
        # 64000 samples a second and a loop would be the most expensive thing
        # in the tick.
        filtered, state = _lfilter()([alpha], [1.0, -(1.0 - alpha)], signal,
                                     zi=[self.low_pass_state[index]])
        self.low_pass_state[index] = float(state[0])
        return filtered.astype(np.float32)


class WindNoise:
    """The noise floor: pink noise, rolled off, breathing at 0.13 Hz.

    THE CHARACTER IS THE PAGE'S. `render3d.html` synthesises its wind from pink
    noise through a bandpass and a low-pass with a slow LFO on the centre
    frequency; the same recipe is here, in the time domain, because what a
    microphone in a gale actually delivers is low-frequency turbulence that
    swells and drops rather than a steady hiss.

    Deterministic: one `numpy.random.Generator` seeded from the run's `--seed`,
    advanced once per microphone per tick. A replay at the same seed hears the
    same gale, which is the only way a table of detection rates means anything.
    """

    def __init__(self, seed=0):
        self.random = np.random.default_rng(seed)
        self.pink_rows = np.zeros((len(MICROPHONE_LAYOUT), 3))
        self.low_pass_state = np.zeros(len(MICROPHONE_LAYOUT))
        self.phase_seconds = 0.0
        self.latest_amplitude = 0.0

    def reset(self) -> None:
        self.pink_rows[:] = 0.0
        self.low_pass_state[:] = 0.0
        self.phase_seconds = 0.0

    def draw(self, microphone_count, sample_count, wind_speed):
        """-> (microphone_count, sample_count) float32 of independent noise."""
        base = wind_noise_amplitude(wind_speed)
        gust = 1.0 + WIND_NOISE_GUST_DEPTH * math.sin(
            2.0 * math.pi * WIND_NOISE_GUST_HZ * self.phase_seconds)
        self.phase_seconds += sample_count / SAMPLE_RATE_HZ
        amplitude = base * gust
        self.latest_amplitude = float(amplitude)
        if amplitude <= 0.0 or sample_count == 0:
            return np.zeros((microphone_count, sample_count), dtype=np.float32)

        white = self.random.standard_normal((microphone_count, sample_count))
        pink = self._pink(white)
        rolled = self._roll_off(pink)
        # Normalised so `amplitude` IS the channel's rms, which is what the law
        # in this module's header promises and what `test_hearing` reports.
        rms = float(np.sqrt(np.mean(rolled ** 2))) or 1.0
        return (rolled / rms * amplitude).astype(np.float32)

    def _pink(self, white):
        """Paul Kellet's three-row pink filter -- the page's own coefficients.

        Written as THREE ONE-POLE FILTERS rather than the page's per-sample
        loop, because they are exactly that and `lfilter` runs them in C. The
        page can afford a JavaScript loop: it fills a three-second buffer once
        and then loops the buffer forever. This runs inside a 20 ms control
        tick, 50 times a second, four channels at a time.
        """
        rows = ((0.99765, 0.0990460), (0.96300, 0.2965164),
                (0.57000, 1.0526913))
        output = white * 0.1848
        for index, (pole, gain) in enumerate(rows):
            filtered, state = _lfilter()(
                [gain], [1.0, -pole], white, axis=-1,
                zi=self.pink_rows[:, index:index + 1])
            self.pink_rows[:, index] = state[:, 0]
            output = output + filtered
        return output * 0.22

    def _roll_off(self, signal):
        alpha = 1.0 - math.exp(-2.0 * math.pi * WIND_NOISE_LOW_PASS_HZ
                               / SAMPLE_RATE_HZ)
        filtered, state = _lfilter()(
            [alpha], [1.0, -(1.0 - alpha)], signal, axis=-1,
            zi=self.low_pass_state[:, None])
        self.low_pass_state[:] = state[:, 0]
        return filtered


# ---------------------------------------------------------------- detector a
class VoiceActivity:
    """webrtcvad -> "a human voice is present", and how much of the window.

    Fed one channel (the front mic). webrtcvad's verdict is BINARY per 30 ms
    frame, so the "probability" reported is the VOICED-FRAME SHARE of a rolling
    480 ms window -- an honest fraction, not a confidence dressed up as one.

    `available` is False and every verdict is False if the library is missing,
    so a machine without it runs the rest of the harness rather than crashing.
    """

    def __init__(self, aggressiveness=VOICE_ACTIVITY_AGGRESSIVENESS):
        self.frame_samples = int(SAMPLE_RATE_HZ
                                 * VOICE_ACTIVITY_FRAME_MILLISECONDS / 1000)
        self.window_frames = max(1, int(VOICE_ACTIVITY_WINDOW_SECONDS
                                        * 1000 / VOICE_ACTIVITY_FRAME_MILLISECONDS))
        self.pending = np.zeros(0, dtype=np.float32)
        self.recent = []
        # The share of the frames consumed by the most recent `feed` -- what the
        # segment machine reads. `probability` is the rolling window, which is
        # what the HUD reads.
        self.latest_share = 0.0
        self.available = False
        self.detector = None
        self.milliseconds = 0.0
        try:
            import webrtcvad
            self.detector = webrtcvad.Vad(int(aggressiveness))
            self.available = True
        except Exception as error:      # pragma: no cover - reporting only
            print(f"[hearing] webrtcvad NOT available ({type(error).__name__}:"
                  f" {error}); voice activity is off and the robot will never"
                  " be called", flush=True)

    def reset(self) -> None:
        self.pending = np.zeros(0, dtype=np.float32)
        self.recent = []
        self.latest_share = 0.0

    def feed(self, channel) -> float:
        """Append audio and consume whole 30 ms frames. -> this call's share.

        Cheap; call at 10 Hz. One 100 ms detector tick is three whole 30 ms
        frames with 10 ms carried to the next call, so the returned share takes
        the values 0, 1/3, 2/3, 1.
        """
        if not self.available:
            return 0.0
        started = time.time()
        self.pending = np.concatenate((self.pending,
                                       np.asarray(channel, dtype=np.float32)))
        verdicts = []
        while self.pending.size >= self.frame_samples:
            frame = self.pending[:self.frame_samples]
            self.pending = self.pending[self.frame_samples:]
            pcm = np.clip(frame * 32767.0, -32768, 32767).astype("<i2").tobytes()
            try:
                voiced = bool(self.detector.is_speech(pcm, SAMPLE_RATE_HZ))
            except Exception:
                voiced = False
            verdicts.append(voiced)
            self.recent.append(voiced)
        if len(self.recent) > self.window_frames:
            self.recent = self.recent[-self.window_frames:]
        self.latest_share = float(np.mean(verdicts)) if verdicts else 0.0
        self.milliseconds = (time.time() - started) * 1000.0
        return self.latest_share

    @property
    def probability(self) -> float:
        """The rolling 480 ms window -- the number the HUD's meter shows."""
        if not self.recent:
            return 0.0
        return float(np.mean(self.recent))

    @property
    def voice_present(self) -> bool:
        """THIS detector tick's verdict. See `VOICE_PRESENT_SHARE`."""
        return self.latest_share >= VOICE_PRESENT_SHARE


# ---------------------------------------------------------------- detector b
class StopWord:
    """Vosk small English, grammar `["stop", "[unk]"]` -> a confidence.

    RUN ONCE PER UTTERANCE, on the whole segment, not per tick. The grammar is
    what makes a 40 MB model usable here: the decoder's search is restricted to
    one word plus an explicit garbage class, so "come here" can only come back
    as `[unk]` and the confidence attached to `stop` is a real posterior over a
    two-way choice rather than an ASR score to be thresholded by feel.

    `available` is False if vosk or its model is missing; the whole hearing
    feature then still runs (VAD and bearing work), and every utterance is
    "voice", never "stop". Said out loud on stdout rather than silently.
    """

    # ONE MODEL PER PROCESS. `test_hearing` builds thousands of these -- one per
    # clip per grid cell -- and a 40 MB acoustic model loaded four thousand
    # times is nine minutes of nothing.
    _shared_model = None

    def __init__(self, verbose=True):
        self.available = False
        self.model = None
        self.milliseconds = 0.0
        try:
            import vosk
            vosk.SetLogLevel(-1)
            started = time.time()
            if StopWord._shared_model is None:
                StopWord._shared_model = vosk.Model(lang="en-us")
            self.model = StopWord._shared_model
            self.available = True
            if verbose:
                print(f"[hearing] vosk small en-us ready in"
                      f" {(time.time() - started) * 1000:.0f} ms, grammar"
                      f" {VOSK_GRAMMAR}", flush=True)
        except Exception as error:      # pragma: no cover - reporting only
            print(f"[hearing] vosk NOT available ({type(error).__name__}:"
                  f" {error}); the word `stop` cannot be recognised. Download"
                  " the model with:  curl -L -o m.zip"
                  " https://alphacephei.com/vosk/models/"
                  "vosk-model-small-en-us-0.15.zip && unzip m.zip -d"
                  " ~/.cache/vosk/", flush=True)

    def confidence(self, segment) -> float:
        """The best `stop` confidence in one utterance. -> 0.0 if not heard."""
        if not self.available:
            return 0.0
        import json
        import vosk

        started = time.time()
        recognizer = vosk.KaldiRecognizer(self.model, SAMPLE_RATE_HZ, VOSK_GRAMMAR)
        recognizer.SetWords(True)
        pcm = np.clip(np.asarray(segment, dtype=np.float32) * 32767.0,
                      -32768, 32767).astype("<i2").tobytes()
        recognizer.AcceptWaveform(pcm)
        result = json.loads(recognizer.FinalResult())
        self.milliseconds = (time.time() - started) * 1000.0
        # THE BEST `stop` ANYWHERE IN THE UTTERANCE, which is what "stop it"
        # needs (it decodes as `[stop 0.96, [unk] 0.88]` and it IS a stop).
        #
        # A "THE UTTERANCE MUST BEGIN WITH THE KEYWORD" rule was tried and
        # DROPPED, and the measurement is worth keeping because the idea is
        # tempting. On one clip it looks decisive -- "top" decodes as
        # `[[unk] 0.62, stop 0.93]`, so requiring the keyword first rejects it.
        # Over the whole 365-clip corpus and the whole grid it moved the
        # near-miss false-stop rate 27.9% -> 27.7% and cost 1.4 points of
        # detection (63.8% -> 62.4%). Most near-misses come back as a BARE
        # `stop`, not as a prefixed one, so the rule was paying for a class of
        # failure that barely exists.
        #
        # The near-miss rate is a real property of a one-word grammar and no
        # threshold fixes it: "top", "shop" and "drop" differ from the keyword
        # by one phoneme, and a 40 MB model asked a two-way question answers it
        # wrong about a quarter of the time at ANY confidence, 1.00 included.
        # What DOES stay at 0.0% is the false-stop rate on ordinary calls --
        # "come here", "over here", "help", and the five demo clips -- which is
        # the population the demo actually contains.
        best = 0.0
        for word in result.get("result", []):
            if word.get("word") == STOP_WORD:
                best = max(best, float(word.get("conf", 0.0)))
        return best


# ---------------------------------------------------------------- detector c
class Bearing:
    """GCC-PHAT across two microphone pairs -> an azimuth in the BODY frame.

    THE ARITHMETIC, and it is the whole reason there are four mics and not two.
    For a source far enough away that the wavefront is flat, the arrival time at
    a microphone at body-frame position `p` is `(R - p.u) / c`, where `u` is the
    unit vector from the robot TOWARD the source. So for an opposed pair,

        tau_ij = t_i - t_j = -(p_i - p_j) . u / c

    Front/back gives `u_x`, left/right gives `u_y`, and the azimuth is
    `atan2(u_y, u_x)`. A SINGLE pair gives one component and therefore a
    front/back (or left/right) ambiguity; two opposed pairs give both components
    and there is no ambiguity left to resolve. Positive = the source is to the
    robot's LEFT, the same sign `guide.StereoEyes` uses and the same sign
    `ang_vel_yaw` wants.

    SUB-SAMPLE OR NOTHING. The 14 cm baseline is 6.8 samples end to end at
    16 kHz, so the correlation peak is interpolated by zero-padding the inverse
    transform `BEARING_INTERPOLATION_FACTOR` times -- the standard GCC-PHAT
    refinement -- which takes the lag resolution to 1/8 sample and the angular
    resolution near broadside to under a degree. Rounded to whole samples the
    array could only ever report about fifteen distinct angles.

    CONFIDENCE is the peak's SHARPNESS: the correlation maximum divided by the
    rms of the correlation over the physically possible lag range, mapped onto
    [0, 1]. A clean utterance gives a single tall spike; a gale gives a lumpy
    field with no spike, and the ratio falls to about 1. It is a measure of
    "is there one direction here", which is exactly the question, and it is not
    a probability of anything.

    Inputs  : `channels` (4, n) -- the EAR signals, never the source.
    Outputs : `(azimuth_radians, confidence)` or `(None, 0.0)`.
    """

    def __init__(self):
        self.milliseconds = 0.0
        self.baseline_meters = 2.0 * MICROPHONE_RADIUS_METERS
        self.maximum_lag_samples = (self.baseline_meters
                                    / SPEED_OF_SOUND_METERS_PER_SECOND
                                    * SAMPLE_RATE_HZ)

    def estimate(self, channels):
        started = time.time()
        channels = np.asarray(channels, dtype=np.float64)
        if channels.shape[1] < 64:
            return None, 0.0
        index = {name: position
                 for position, name in enumerate(MICROPHONE_NAMES)}
        forward_lag, forward_sharpness = self._pair(
            channels[index["front"]], channels[index["back"]])
        lateral_lag, lateral_sharpness = self._pair(
            channels[index["left"]], channels[index["right"]])
        self.milliseconds = (time.time() - started) * 1000.0
        if forward_lag is None or lateral_lag is None:
            return None, 0.0
        scale = SPEED_OF_SOUND_METERS_PER_SECOND / (
            self.baseline_meters * SAMPLE_RATE_HZ)
        forward_component = -forward_lag * scale
        lateral_component = -lateral_lag * scale
        norm = math.hypot(forward_component, lateral_component)
        if norm < 1e-6:
            return None, 0.0
        azimuth = math.atan2(lateral_component, forward_component)
        # Both pairs have to agree that there is a peak; the weaker one is the
        # one that decides, because a bearing is only as good as its worse axis.
        sharpness = min(forward_sharpness, lateral_sharpness)
        confidence = float(np.clip((sharpness - BEARING_SHARPNESS_FLOOR)
                                   / BEARING_SHARPNESS_SPAN, 0.0, 1.0))
        return float(azimuth), confidence

    def _pair(self, first, second):
        """GCC-PHAT lag of `first` relative to `second`. -> (samples, sharpness)."""
        count = first.size
        length = 1
        while length < 2 * count:
            length *= 2
        first_spectrum = np.fft.rfft(first - first.mean(), n=length)
        second_spectrum = np.fft.rfft(second - second.mean(), n=length)
        cross = first_spectrum * np.conj(second_spectrum)
        magnitude = np.abs(cross)
        if magnitude.max() <= 0.0:
            return None, 0.0
        # PHASE TRANSFORM: divide out the magnitude and keep only the phase, so
        # the estimate depends on WHEN things happened and not on how loud they
        # were. That is what makes it robust to the voice's own spectrum and to
        # a wind noise that is all low frequency.
        #
        # REGULARISED, and it is not a nicety. A textbook PHAT divides by the
        # magnitude everywhere, which multiplies EMPTY bins -- bins holding
        # nothing but floating-point dust -- up to unit magnitude with garbage
        # phase, and then averages that garbage into the peak. On a narrowband
        # source (a pure tone, or a vowel) most bins are empty and the estimate
        # falls apart: MEASURED on a 700 Hz tone at +30 degrees, the unfloored
        # version answered +6 degrees. Flooring the divisor at
        # `PHASE_TRANSFORM_FLOOR` of the strongest bin leaves the loud bins
        # fully whitened and leaves the dust where it was.
        cross /= np.maximum(magnitude, PHASE_TRANSFORM_FLOOR * magnitude.max())
        upsampled = length * BEARING_INTERPOLATION_FACTOR
        correlation = np.fft.irfft(cross, n=upsampled)
        span = int(math.ceil(self.maximum_lag_samples
                             * BEARING_INTERPOLATION_FACTOR)) + 2
        window = np.concatenate((correlation[-span:], correlation[:span + 1]))
        lags = np.arange(-span, span + 1) / BEARING_INTERPOLATION_FACTOR
        peak = int(np.argmax(window))
        lag = float(np.clip(lags[peak], -self.maximum_lag_samples,
                            self.maximum_lag_samples))
        # THE FLOOR IS MEASURED WELL OUTSIDE THE PHYSICAL LAG RANGE, and that is
        # not arbitrary. The array's whole lag range is +/-6.8 samples, so a
        # peak-to-rms taken over that range is mostly the peak dividing itself
        # and the ratio saturates: MEASURED, it read 0.25 for every cell of the
        # grid from a flat calm to a gale, which is a number reading the dial
        # rather than the world. Taken over a lag range the source CANNOT
        # occupy, the denominator is the correlator's genuine background and the
        # ratio moves.
        wide = int(self.maximum_lag_samples * BEARING_INTERPOLATION_FACTOR
                   * BEARING_FLOOR_SPAN_MULTIPLIER)
        background = np.concatenate((correlation[span + 1:wide],
                                     correlation[-wide:-span]))
        floor = float(np.sqrt(np.mean(background ** 2))) if background.size else 0.0
        sharpness = float(window[peak] / (floor or 1e-12))
        return lag, sharpness


# ------------------------------------------------------- the whole sensor
class Ears:
    """The four microphones and the three detectors, driven once per tick.

    Inputs  : this tick's microphone samples, the mouth's world position, the
              robot head's world frame, and the wind speed.
    Outputs : `channels` (4, n) for the tick, and -- at
              `DETECTOR_EVERY_N_TICKS` -- an updated `voice_probability`,
              `heard`, `stop_confidence`, `bearing_radians`,
              `bearing_confidence` and `level_db`.
    """

    def __init__(self, seed=0, verbose=True, control_hz=50.0):
        self.wind_noise = WindNoise(seed=seed)
        self.array = EarArray(wind_noise=self.wind_noise)
        self.voice_activity = VoiceActivity()
        self.stop_word = StopWord(verbose=verbose)
        self.bearing_estimator = Bearing()
        self.control_hz = float(control_hz)
        self.samples_per_tick = int(round(SAMPLE_RATE_HZ / self.control_hz))

        self.voice_probability = 0.0
        self.level_db = -80.0
        self.heard = "none"
        self.stop_confidence = 0.0
        self.bearing_radians = None
        self.bearing_confidence = 0.0
        self.segments_heard = 0
        self.new_segment = False        # True for exactly one tick per utterance
        # HOW LONG AGO THE BEARING'S OWN AUDIO HAPPENED. A bearing is measured
        # on the loudest 300 ms of an utterance, and the utterance is only
        # decided once it has ENDED -- so by the time a cue is published, the
        # sound it was measured from is up to a second old, and the robot may
        # have turned in the meantime. Anything that converts this bearing into
        # a world heading must use the yaw the robot had THEN, and this is the
        # number that lets it. MEASURED on terrain_free_5, where the stock
        # walker swings its yaw about 50 deg/s at zero command: using the
        # current yaw instead put the cue 25 to 60 degrees out.
        self.bearing_age_seconds = 0.0
        # The loudest the rolling VAD window got during the last utterance --
        # reported by `test_hearing` so a miss can be told from a near-miss.
        self.segment_peak_voice_probability = 0.0
        # The loudest the front ear got during the last utterance, dBFS.
        self.segment_peak_level_db = -80.0
        self._segment_peak = 0.0
        self._segment_peak_level = -80.0

        self._segment = []
        self._segment_channels = []
        self._silence_seconds = 0.0
        self._segment_seconds = 0.0
        self._detector_buffer = []      # channels awaiting a detector tick
        self._preroll = []              # the last few blocks, for the onset
        self.array_milliseconds = 0.0
        self.detector_milliseconds = 0.0
        if verbose:
            print(describe_wind_law(), flush=True)
            print(f"[hearing] {len(MICROPHONE_LAYOUT)} mics at"
                  f" {MICROPHONE_RADIUS_METERS * 100:.0f} cm"
                  f" ({', '.join(MICROPHONE_NAMES)}), c ="
                  f" {SPEED_OF_SOUND_METERS_PER_SECOND:.0f} m/s ->"
                  f" {2 * MICROPHONE_RADIUS_METERS / SPEED_OF_SOUND_METERS_PER_SECOND * 1e6:.0f} us"
                  f" ({2 * MICROPHONE_RADIUS_METERS / SPEED_OF_SOUND_METERS_PER_SECOND * SAMPLE_RATE_HZ:.1f}"
                  f" samples) end to end; detectors at"
                  f" {self.control_hz / DETECTOR_EVERY_N_TICKS:.0f} Hz",
                  flush=True)

    def reset(self) -> None:
        self.array.reset()
        self.wind_noise.reset()
        self.voice_activity.reset()
        self.voice_probability = 0.0
        self.level_db = -80.0
        self.heard = "none"
        self.stop_confidence = 0.0
        self.bearing_radians = None
        self.bearing_confidence = 0.0
        self.new_segment = False
        self.bearing_age_seconds = 0.0
        self.segment_peak_voice_probability = 0.0
        # The loudest the front ear got during the last utterance, dBFS.
        self.segment_peak_level_db = -80.0
        self._segment_peak = 0.0
        self._segment_peak_level = -80.0
        self._segment = []
        self._segment_channels = []
        self._silence_seconds = 0.0
        self._segment_seconds = 0.0
        self._detector_buffer = []
        self._preroll = []

    def microphone_positions_world(self, head_position, head_rotation):
        """The four capsules in the world. -> (4, 3)."""
        offsets = np.array([direction for _, direction in MICROPHONE_LAYOUT],
                           dtype=float) * MICROPHONE_RADIUS_METERS
        return np.asarray(head_position, dtype=float) + offsets @ np.asarray(
            head_rotation, dtype=float).T

    def update(self, samples, mouth_position_world, head_position_world,
               head_rotation_world, wind_speed_meters_per_second, tick):
        """One control tick. -> the (4, n) ear channels."""
        started = time.time()
        microphones = self.microphone_positions_world(head_position_world,
                                                      head_rotation_world)
        ranges = np.linalg.norm(
            microphones - np.asarray(mouth_position_world, dtype=float), axis=1)
        channels = self.array.feed(samples, ranges, wind_speed_meters_per_second)
        self.array_milliseconds = (time.time() - started) * 1000.0
        self._detector_buffer.append(channels)
        self.new_segment = False
        if tick % DETECTOR_EVERY_N_TICKS == 0:
            self._run_detectors()
        return channels

    def _run_detectors(self) -> None:
        started = time.time()
        if not self._detector_buffer:
            self.detector_milliseconds = 0.0
            return
        block = np.concatenate(self._detector_buffer, axis=1)
        self._detector_buffer = []
        monitor = block[MICROPHONE_NAMES.index(MONITOR_MICROPHONE)]
        self.level_db = decibels(float(np.sqrt(np.mean(monitor ** 2))))

        self.voice_activity.feed(monitor)
        self.voice_probability = self.voice_activity.probability
        self._segment_peak = max(self._segment_peak, self.voice_probability)
        self._segment_peak_level = max(self._segment_peak_level, self.level_db)
        voiced = self.voice_activity.voice_present
        seconds = block.shape[1] / SAMPLE_RATE_HZ

        if voiced:
            if not self._segment:
                # THE ONSET. Glue the pre-roll on before the first voiced block,
                # so the recogniser gets the whole word and not its tail.
                for past_monitor, past_block in self._preroll:
                    self._segment.append(past_monitor)
                    self._segment_channels.append(past_block)
                    self._segment_seconds += past_block.shape[1] / SAMPLE_RATE_HZ
            self._segment.append(monitor)
            self._segment_channels.append(block)
            self._segment_seconds += seconds
            self._silence_seconds = 0.0
        elif self._segment:
            # Keep the trailing silence IN the segment: vosk wants the release
            # of the final plosive, and "stop" ends in one.
            self._segment.append(monitor)
            self._segment_channels.append(block)
            self._segment_seconds += seconds
            self._silence_seconds += seconds
        # The pre-roll ring is kept whether or not a segment is open; the
        # blocks it holds cost 100 ms each and it is three deep.
        self._preroll.append((monitor, block))
        preroll_blocks = max(1, int(round(SEGMENT_PREROLL_SECONDS
                                          / max(seconds, 1e-6))))
        while len(self._preroll) > preroll_blocks:
            self._preroll.pop(0)

        if self._segment and (self._silence_seconds >= SEGMENT_END_SILENCE_SECONDS
                              or self._segment_seconds >= SEGMENT_MAXIMUM_SECONDS):
            self._close_segment()
        self.detector_milliseconds = (time.time() - started) * 1000.0

    def _close_segment(self) -> None:
        """One utterance is over: decide the word and the direction."""
        monitor = np.concatenate(self._segment)
        channels = np.concatenate(self._segment_channels, axis=1)
        self._segment = []
        self._segment_channels = []
        # THE PRE-ROLL RING IS NOT CLEARED. It now holds the trailing silence
        # this segment closed on, which is exactly the right context for the
        # next utterance -- clearing it would leave a second shout, a second
        # after the first, with no onset again.
        speech_seconds = self._segment_seconds - self._silence_seconds
        self._segment_seconds = 0.0
        self._silence_seconds = 0.0
        peak, self._segment_peak = self._segment_peak, 0.0
        peak_level, self._segment_peak_level = self._segment_peak_level, -80.0
        if speech_seconds < SEGMENT_MINIMUM_SECONDS:
            return
        self.segment_peak_voice_probability = peak
        self.segment_peak_level_db = peak_level
        self.segments_heard += 1
        self.new_segment = True
        self.stop_confidence = self.stop_word.confidence(monitor)
        self.heard = ("stop" if self.stop_confidence >= STOP_CONFIDENCE_THRESHOLD
                      else "voice")
        window, window_start = self._loudest_window(channels)
        # The centre of the window the bearing came from, counted back from the
        # end of the utterance (which is NOW).
        self.bearing_age_seconds = float(
            (channels.shape[1] - (window_start + window.shape[1] / 2.0))
            / SAMPLE_RATE_HZ)
        azimuth, confidence = self.bearing_estimator.estimate(window)
        if azimuth is not None:
            self.bearing_radians = azimuth
            self.bearing_confidence = confidence
        else:
            self.bearing_confidence = 0.0

    @staticmethod
    def _loudest_window(channels):
        """The loudest `BEARING_WINDOW_SECONDS` of the utterance.
        -> ((4, m), start_index).

        The head and tail of an utterance are mostly noise floor, and
        cross-correlating the noise floor with itself is how a bearing table
        comes out flat.
        """
        width = int(BEARING_WINDOW_SECONDS * SAMPLE_RATE_HZ)
        total = channels.shape[1]
        if total <= width:
            return channels, 0
        energy = channels[0] ** 2
        cumulative = np.concatenate(([0.0], np.cumsum(energy)))
        sums = cumulative[width:] - cumulative[:-width]
        start = int(np.argmax(sums))
        return channels[:, start:start + width], start

    def timing(self) -> dict:
        return {
            "array_milliseconds": self.array_milliseconds,
            "detector_milliseconds": self.detector_milliseconds,
            "voice_activity_milliseconds": self.voice_activity.milliseconds,
            "stop_word_milliseconds": self.stop_word.milliseconds,
            "bearing_milliseconds": self.bearing_estimator.milliseconds,
        }


# ------------------------------------------------------------- the behaviour
class HearingBehaviour:
    """What the robot DOES about a voice. A layer over `guide.GuideFollower`.

    THE RULES, and they are the user's, verbatim in behaviour:

      * a confident `stop` -> `STOPPED`. The robot stands, and stays stopped.
      * any OTHER human voice -> the robot COMES TO HER. If the eyes can see
        her, the existing FOLLOW/WAIT does the navigation and stops at 1 m
        (`COMING_BY_EYES`). If they cannot, the ear bearing does
        (`COMING_BY_EARS`), re-estimated on every new utterance, until the eyes
        acquire and vision takes over.
      * VOICE IS A TRIGGER, NOT A LEASH. One shout starts the walk; silence does
        not stop it. It ends at the person (`WAIT`) or at a `stop`.
      * LOSING HER MID-WALK IS NOT A SEARCH ANY MORE (user's ruling,
        2026-08-30). The follower's camera sweep is deleted. Called, with no
        eyes on her and no ear cue worth steering by, the robot STANDS STILL AND
        LISTENS (`LISTENING`) -- and the next shout hands it a direction, which
        is a thing a camera sweep could never do. That is the whole argument for
        ears: a lost robot should wait to be called, not wave its torso about.

    WHY IDLE AND LISTENING BOTH COMMAND ZERO, and why they are still two states.
    `IDLE` is "nobody has called me yet"; `LISTENING` is "I was called, I have
    lost her, and I do not know which way to go". The command is the same and
    the situation is not, and a demo in which those two look identical is a demo
    nobody can read.

    With hearing OFF this class is never consulted and the guide follower's
    command goes through untouched.

    Inputs  : the `Ears` (its verdicts only), and the `GuideFollower`'s live
              state.
    Outputs : `mode` (one of `HEARING_MODES`) and `command()` -> (3,).
    """

    def __init__(self, walk_speed=EAR_WALK_SPEED_METERS_PER_SECOND):
        self.walk_speed = float(walk_speed)
        self.mode = "IDLE"
        self.stopped = False
        self.called = False
        # THE CUE IS STORED AS A WORLD HEADING, NOT AS A BODY BEARING, and this
        # is the single most important line in the class. The ears measure a
        # direction relative to the ROBOT; the robot then turns, which makes
        # that number stale immediately. Steering by the stored body bearing is
        # therefore a positive feedback loop dressed as a controller: the robot
        # keeps commanding the same +31 degrees whatever it is now facing and
        # walks a circle. MEASURED before this was fixed -- on terrain_free_5
        # the robot spiralled for eleven seconds and fell over
        # (`tipped_over`, upright -0.08), having got no closer than 2.7 m to a
        # person 6 m away. Converting the bearing to a world heading ONCE, at
        # the moment it is measured, and re-deriving the body-frame error from
        # the current yaw every tick, is the whole fix.
        self.cue_heading_world_radians = None
        self.cue_bearing_radians = None      # as measured, for the HUD
        self.cue_confidence = 0.0
        self.cue_age_seconds = 0.0
        # A cue is SPENT once the eyes have taken over from it. What that buys:
        # the robot walks to her on one shout (silence does not stop it), and
        # yet losing her mid-walk AFTER vision had her does not send it
        # trudging off along a bearing that was true a minute ago -- it stands
        # and LISTENS, which is the user's 2026-08-30 ruling.
        self.cue_spent = False
        self.bearing_error_radians = 0.0     # what the yaw command is closing
        # WALK THE VECTOR (user's ruling, 2026-08-30). Not a constructor
        # argument on purpose: the FOLLOWER already knows whether this world
        # lets the body move sideways (`GuideFollower.vector_steering`, set from
        # the rope flag), and it is handed to `update` on every tick, so reading
        # it there keeps ears and eyes on ONE steering law with no second place
        # to configure and get wrong.
        self.vector_steering = False
        self._command = np.zeros(3)

    def reset(self) -> None:
        self.mode = "IDLE"
        self.stopped = False
        self.called = False
        self.cue_heading_world_radians = None
        self.cue_bearing_radians = None
        self.cue_confidence = 0.0
        self.cue_age_seconds = 0.0
        self.cue_spent = False
        self.bearing_error_radians = 0.0
        self._command = np.zeros(3)

    @staticmethod
    def eyes_have_her(follower) -> bool:
        """Is the VISION follower currently holding a live measurement?"""
        from app.harness import guide as guide_module
        return (follower is not None
                and follower.range_meters is not None
                and follower.seconds_since_detection
                < guide_module.LOST_AFTER_SECONDS
                and follower.mode in ("FOLLOW", "WAIT"))

    def update(self, ears: "Ears", follower, guide_command, dt_seconds: float,
               robot_yaw_radians: float = 0.0,
               measurement_yaw_radians: float = None):
        """One control tick. -> the (3,) command to fly.

        `robot_yaw_radians` is the robot's heading about world +z NOW, and
        `measurement_yaw_radians` is the heading it had when the bearing's own
        audio arrived (`Ears.bearing_age_seconds` ago). The cue is converted
        with the second and closed with the first. See the comment on
        `cue_heading_world_radians`.
        """
        if measurement_yaw_radians is None:
            measurement_yaw_radians = robot_yaw_radians
        self.vector_steering = bool(getattr(follower, "vector_steering", False))
        self.cue_age_seconds += dt_seconds

        if ears.new_segment:
            if ears.heard == "stop":
                self.stopped = True
                self.called = False
            else:
                self.called = True
                self.stopped = False
                if (ears.bearing_radians is not None
                        and ears.bearing_confidence >= EAR_BEARING_MINIMUM_CONFIDENCE):
                    self.cue_bearing_radians = float(ears.bearing_radians)
                    self.cue_heading_world_radians = _wrap_to_pi(
                        float(measurement_yaw_radians)
                        + float(ears.bearing_radians))
                    self.cue_confidence = float(ears.bearing_confidence)
                    self.cue_age_seconds = 0.0
                    self.cue_spent = False

        if self.stopped:
            self.mode = "STOPPED"
            self._command = np.zeros(3)
            return self._command

        if not self.called:
            # NOBODY HAS CALLED YET (or the last call ended at her heels). With
            # ears ON BY DEFAULT (user ruling 2026-08-30) this state must not
            # stomp the follower: the guide command passes through untouched,
            # so vision and the W gate drive exactly as if hearing were off.
            # Zeroing here -- fine when hearing was opt-in -- stopped the robot
            # dead at boot and dropped it on slopes (user-found regression).
            self.mode = "IDLE"
            self._command = (np.asarray(guide_command, dtype=float).copy()
                             if guide_command is not None else np.zeros(3))
            return self._command

        if self.eyes_have_her(follower):
            # VISION OWNS THE NAVIGATION from here. The follower's own command
            # is handed through byte for byte -- its hysteresis and its 1 m WAIT
            # band are exactly what they were.
            self.cue_spent = True         # vision has it from here
            if follower.mode == "WAIT":
                self.mode = "WAIT"
                self.called = False       # arrived; the next shout re-calls
            else:
                self.mode = "COMING_BY_EYES"
            self._command = np.asarray(guide_command, dtype=float).copy()
            return self._command

        if (self.cue_heading_world_radians is None or self.cue_spent
                or self.cue_age_seconds > EAR_CUE_VALID_SECONDS):
            # Called, but with no eyes on her and no direction worth steering
            # by. STAND STILL AND LISTEN. Walking off in the last known
            # direction would be worse than useless: the cue is stale precisely
            # because she has not spoken for twenty seconds, and a robot
            # wandering away is a robot that cannot be called back.
            self.mode = "LISTENING"
            self.bearing_error_radians = 0.0
            self._command = np.zeros(3)
            return self._command

        self.mode = "COMING_BY_EARS"
        self.bearing_error_radians = _wrap_to_pi(
            self.cue_heading_world_radians - float(robot_yaw_radians))
        self._command = self._walk_toward(self.bearing_error_radians)
        return self._command

    def _walk_toward(self, bearing_radians) -> np.ndarray:
        """Walk, turning toward a body-frame bearing. -> (3,).

        THE ROBOT NEVER TURNS ON THE SPOT, and that reads backwards until you
        see the number. The obvious design is "more than 60 degrees off? stop,
        pivot, then walk", because walking while turning traces a long arc. It
        does not work: THIS POLICY HAS NO YAW AUTHORITY AT ZERO FORWARD SPEED,
        because yaw comes out of the stepping gait and a robot commanded
        `[0, 0, +0.5]` is standing still. MEASURED on `terrain_free_0` with the
        pivot-first rule, the heading error sat at +80 degrees for EIGHTY-FIVE
        SECONDS, the waist pinned at its limit, the base not moving, and the run
        ended further from her than it started. Walking throughout, the same
        controller reaches a person 6 m away on `flat_free`. The arc is the
        price and it is worth paying.
        """
        bearing = float(bearing_radians)
        if self.vector_steering:
            # WALK THE VECTOR. `lin_vel_y` carries the sideways part of the
            # approach directly instead of asking a 7%-authority yaw port to
            # rotate the whole robot first -- `guide.vector_command` holds the
            # measured table. The "never turn on the spot" finding below still
            # governs: `lin_vel_x` is floored at -0.2 m/s rather than zeroed, so
            # the gait keeps stepping and the yaw term keeps working even when
            # she is directly behind.
            from app.harness import guide as guide_module
            return guide_module.vector_command(
                bearing, self.walk_speed,
                deadband_radians=EAR_BEARING_DEADBAND_RADIANS)
        if abs(bearing) < EAR_BEARING_DEADBAND_RADIANS:
            yaw_rate = 0.0
        else:
            yaw_rate = float(np.clip(
                EAR_BEARING_GAIN_PER_RADIAN * bearing,
                -EAR_MAXIMUM_YAW_RATE_RADIANS_PER_SECOND,
                EAR_MAXIMUM_YAW_RATE_RADIANS_PER_SECOND))
        return np.array([self.walk_speed, 0.0, yaw_rate])

    def command(self) -> np.ndarray:
        return self._command


# ------------------------------------------------------- the whole feature
class HearingSystem:
    """Ears + behaviour, wired together and driven by one call a tick.

    This is what `runtime.run` holds. It owns the microphone ring, the truth
    geometry lookups, the `EAR0` monitor mix, the state block and the recorded
    columns, and it is a no-op in every entry point when the knob is off.

    THE ONE THING IT NEVER DOES is write to `MjData` or to the model. It reads
    `data.xpos` and `data.xmat` for the robot's head and asks the guide where
    its mouth is; nothing else. `test_hearing` section 5 is that claim measured.
    """

    def __init__(self, model, control_hz, seed=0, verbose=True,
                 walk_speed=EAR_WALK_SPEED_METERS_PER_SECOND):
        self.model = model
        self.control_hz = float(control_hz)
        self.dt_seconds = 1.0 / self.control_hz
        self.microphone = MicrophoneStream()
        self.ears = Ears(seed=seed, verbose=verbose, control_hz=self.control_hz)
        self.behaviour = HearingBehaviour(walk_speed=walk_speed)
        self.enabled = False
        self.monitor_enabled = False
        self.ear_pcm = None
        self.injectors = []
        self.samples_per_tick = self.ears.samples_per_tick
        # (time_seconds, torso_yaw) for the last few seconds. See
        # `Ears.bearing_age_seconds` for why a cue needs the OLD yaw, and
        # `update` for why it is the TORSO's yaw and not the base's.
        self._yaw_history = []
        self._monitor_pending = []
        self.head_body_id = self._find_head_body(model, verbose=verbose)

    @staticmethod
    def _find_head_body(model, verbose=True) -> int:
        import mujoco
        for name in HEAD_BODY_NAMES:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                if verbose:
                    print(f"[hearing] microphone array rides body {name!r}"
                          f" (id {body_id}): +x forward, +y left -- the same"
                          " body the eyes are mounted on, so an ear bearing and"
                          " an image bearing mean the same angle", flush=True)
                return body_id
        print(f"[hearing] none of {HEAD_BODY_NAMES} is in this model; the ears"
              " fall back to body 1 and the bearing frame is unverified",
              flush=True)
        return 1

    def add_injector(self, path: str, start_seconds: float) -> None:
        self.injectors.append(VoiceInjector(path, start_seconds))
        print(f"[hearing] --inject-voice {path} at t = {start_seconds:.2f} s"
              f" ({self.injectors[-1].samples.size / SAMPLE_RATE_HZ:.2f} s of"
              " audio, pushed into the SAME ring the browser pushes into)",
              flush=True)

    def bind(self, model) -> None:
        """A map switch is a new compiled model; re-look-up the head body."""
        self.model = model
        self.head_body_id = self._find_head_body(model, verbose=False)

    def reset(self) -> None:
        self.ears.reset()
        self.behaviour.reset()
        self.microphone.clear()
        self.ear_pcm = None
        self._monitor_pending = []
        self._yaw_history = []

    def _yaw_at(self, time_seconds: float) -> float:
        """The robot's yaw at (or nearest to) a time in the recent past."""
        if not self._yaw_history:
            return 0.0
        best = min(self._yaw_history,
                   key=lambda row: abs(row[0] - float(time_seconds)))
        return float(best[1])

    # ------------------------------------------------------------- geometry
    def head_frame(self, data):
        """-> (position (3,), rotation (3,3)) of the microphone array's body."""
        position = np.asarray(data.xpos[self.head_body_id], dtype=float)
        rotation = np.asarray(data.xmat[self.head_body_id],
                              dtype=float).reshape(3, 3)
        return position, rotation

    @staticmethod
    def mouth_position_world(guide_system):
        """Where the voice comes OUT: the hiker's mouth. -> (3,) world.

        Computed from her root pose and the head geom's offset in her own frame
        rather than read out of `geom_xpos`, because the guide restores the
        frames it refreshed and `geom_xpos` is therefore one step stale. Her
        head hangs off the MOCAP ROOT (not off one of the six swinging limbs),
        so root + Rz(yaw) . head_offset is exact, not an approximation.
        """
        from app.harness import guide as guide_module

        guide = guide_system.guide
        root = guide.root_world()
        yaw = guide.yaw_radians()
        offset = _mouth_offset_in_guide_frame()
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return np.array([
            root[0] + cosine * offset[0] - sine * offset[1],
            root[1] + sine * offset[0] + cosine * offset[1],
            root[2] + offset[2],
        ])

    # --------------------------------------------------------------- the tick
    def update(self, data, tick, enabled, guide_system, guide_command,
               wind_speed_meters_per_second, time_seconds):
        """One control tick. -> the command to fly, or None if hearing is off.

        `guide_command` is whatever `GuideSystem.update` just returned (None if
        the guide is off). Hearing needs the guide: the mouth is on the guide's
        head and the eyes are the guide follower's. With the guide off, hearing
        reports itself disabled and changes nothing.
        """
        was_enabled, self.enabled = self.enabled, bool(
            enabled and guide_system is not None and guide_system.available
            and guide_system.enabled)
        if not self.enabled:
            if was_enabled:
                self.reset()
            return None
        if not was_enabled:
            self.reset()

        for injector in self.injectors:
            injector.step(time_seconds, self.samples_per_tick, self.microphone)
        samples = self.microphone.take(self.samples_per_tick)

        head_position, head_rotation = self.head_frame(data)
        mouth = self.mouth_position_world(guide_system)
        channels = self.ears.update(samples, mouth, head_position,
                                    head_rotation, wind_speed_meters_per_second,
                                    tick)
        if self.monitor_enabled:
            self._monitor_pending.append(
                channels[MICROPHONE_NAMES.index(MONITOR_MICROPHONE)])
            if tick % DETECTOR_EVERY_N_TICKS == 0:
                mix = np.concatenate(self._monitor_pending)
                self._monitor_pending = []
                self.ear_pcm = EAR_MESSAGE_PREFIX + np.clip(
                    mix * 32767.0, -32768, 32767).astype("<i2").tobytes()
        follower = guide_system.follower if guide_system is not None else None
        # TWO YAWS, AND THEY ARE NOT THE SAME ANGLE. The ear cue is remembered
        # in the WORLD frame (`HearingBehaviour.cue_heading_world_radians`), so
        # the behaviour needs both of these:
        #   base yaw   -- the free joint's heading. This is what `ang_vel_yaw`
        #                 steers, so it is what a heading ERROR must be measured
        #                 against.
        #   torso yaw  -- the microphone array's own heading. The ear bearing is
        #                 measured in THIS frame, so it is what a bearing must be
        #                 converted to a world heading WITH.
        # They differ by the waist joint plus whatever the torso is doing, and on
        # this robot that is not small: MEASURED at the spawn of
        # `terrain_free_0`, a hiker placed 45 deg off the BASE sits 24 deg off
        # the TORSO. Converting a torso-frame bearing with the base's yaw put
        # every ear cue about 20 degrees wide, and the robot walked past her.
        # The measurement yaw is looked up `Ears.bearing_age_seconds` in the
        # past, because that is when the sound the bearing came from arrived.
        w, x, y, z = [float(v) for v in data.qpos[3:7]]
        robot_yaw = math.atan2(2.0 * (w * z + x * y),
                               1.0 - 2.0 * (y * y + z * z))
        rotation = np.asarray(data.xmat[self.head_body_id],
                              dtype=float).reshape(3, 3)
        torso_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        self._yaw_history.append((float(time_seconds), torso_yaw))
        horizon = float(time_seconds) - YAW_HISTORY_SECONDS
        while self._yaw_history and self._yaw_history[0][0] < horizon:
            self._yaw_history.pop(0)
        measurement_yaw = self._yaw_at(
            float(time_seconds) - self.ears.bearing_age_seconds)
        command = self.behaviour.update(self.ears, follower, guide_command,
                                        self.dt_seconds, robot_yaw,
                                        measurement_yaw)
        self._aim_waist(guide_system)
        return command

    def _aim_waist(self, guide_system) -> None:
        """Point the stereo pair at the shout while the cue is fresh.

        The waist is written AFTER `GuideSystem.update` has advanced it, so a
        target set here lands on the next tick -- 20 ms, which nothing can
        notice. It deliberately does NOT rate-limit or clamp anything itself:
        `guide.WaistYaw` owns both, and duplicating them here is how two
        clamps end up disagreeing.

        Letting go of the waist once the cue goes stale matters more than it
        looks: a wrong bearing held for ever is a robot staring at an empty
        slope with its torso twisted, and the waist limit is 60 degrees for a
        measured reason (`guide.WAIST_LIMIT_RADIANS`).
        """
        behaviour = self.behaviour
        if (guide_system is None or getattr(guide_system, "waist", None) is None
                or behaviour.mode != "COMING_BY_EARS"
                or behaviour.cue_bearing_radians is None
                or behaviour.cue_age_seconds > EAR_WAIST_AIM_SECONDS):
            return
        # The cue in the BODY's frame right now: the remembered world heading
        # minus where the body is pointing. `bearing_error_radians` is already
        # exactly that.
        guide_system.waist.target_radians = float(np.clip(
            behaviour.bearing_error_radians,
            -EAR_WAIST_AIM_LIMIT_RADIANS, EAR_WAIST_AIM_LIMIT_RADIANS))

    def take_ear_pcm(self):
        """The newest monitor mix, ONCE. -> bytes or None."""
        pcm, self.ear_pcm = self.ear_pcm, None
        return pcm

    # ---------------------------------------------------------------- output
    def state(self) -> dict:
        """The `hearing` block of the websocket state message."""
        ears = self.ears
        bearing = ears.bearing_radians
        return {
            "enabled": bool(self.enabled),
            "voice_probability": round(float(ears.voice_probability), 3),
            # The verdict on the LAST utterance: "none" until one is heard.
            "heard": ears.heard,
            "stop_confidence": round(float(ears.stop_confidence), 3),
            "bearing_degrees": (None if bearing is None
                                else round(math.degrees(bearing), 1)),
            "bearing_confidence": round(float(ears.bearing_confidence), 3),
            "mode": self.behaviour.mode,
            # What the EAR hears (front microphone, after 1/r and the wind).
            "level_db": round(float(ears.level_db), 1),
            # What the MICROPHONE delivered, raw and un-normalised -- the meter
            # that shows the operator their own shout.
            "microphone_level_db": round(decibels(
                self.microphone.recent_rms), 1),
            "microphone_peak_db": round(decibels(
                self.microphone.recent_peak), 1),
        }

    def recorded(self) -> dict:
        """The columns `Recorder` stacks -- all floats, like the guide's."""
        ears = self.ears
        bearing = ears.bearing_radians
        return {
            "hearing_mode": HEARING_MODE_CODES.get(self.behaviour.mode, -1.0),
            "hearing_enabled": 1.0 if self.enabled else 0.0,
            "hearing_voice_probability": float(ears.voice_probability),
            "hearing_heard": HEARD_CODES.get(ears.heard, -1.0),
            "hearing_stop_confidence": float(ears.stop_confidence),
            "hearing_bearing_degrees": (NO_MEASUREMENT if bearing is None
                                        else float(math.degrees(bearing))),
            "hearing_bearing_confidence": float(ears.bearing_confidence),
            "hearing_level_db": float(ears.level_db),
        }

    def describe_cost(self) -> str:
        timing = self.ears.timing()
        return (f"[hearing] per tick: array {timing['array_milliseconds']:.3f} ms"
                f" | detectors {timing['detector_milliseconds']:.3f} ms"
                f" (vad {timing['voice_activity_milliseconds']:.3f},"
                f" vosk {timing['stop_word_milliseconds']:.1f} per utterance,"
                f" gcc-phat {timing['bearing_milliseconds']:.2f} per utterance)")


_MOUTH_OFFSET = None


def _mouth_offset_in_guide_frame() -> np.ndarray:
    """The mouth, in the hiker's own frame. -> (3,) metres, cached.

    Read off HER geometry rather than typed: the `human_head` geom's centre in
    the root frame, pushed forward by the head's own radius so the sound leaves
    the FACE rather than the middle of the skull. Falls back to a plain height
    if her head is ever renamed, and says so.
    """
    global _MOUTH_OFFSET
    if _MOUTH_OFFSET is not None:
        return _MOUTH_OFFSET
    from app.harness import guide as guide_module

    head = None
    for geom in guide_module.guide_skeleton()["root_geoms"]:
        if geom["name"] == "human_head":
            head = geom
            break
    if head is None:
        print("[hearing] no `human_head` geom on the guide; the mouth falls"
              " back to 1.60 m on her body axis", flush=True)
        _MOUTH_OFFSET = np.array([0.0, 0.0, 1.60])
        return _MOUTH_OFFSET
    radius = float(np.asarray(head["size"], dtype=float)[0])
    position = np.asarray(head["pos"], dtype=float)
    _MOUTH_OFFSET = np.array([position[0] + radius, position[1], position[2]])
    return _MOUTH_OFFSET


if __name__ == "__main__":
    # Distributional sanity, printed rather than asserted: an ear model that
    # looks plausible in code and is broken in numbers is the easiest thing in
    # the world to ship.
    print(describe_wind_law())
    array = EarArray(wind_noise=WindNoise(seed=0))
    estimator = Bearing()
    random = np.random.default_rng(0)
    sources = {
        # Broadband is what an array wants and what speech mostly is.
        "broadband noise": random.standard_normal(9600).astype(np.float32) * 0.3,
        # A single vowel-like tone is the worst case for a phase transform, and
        # the reason `PHASE_TRANSFORM_FLOOR` exists.
        "700 Hz tone": (np.sin(2 * np.pi * 700
                               * np.arange(9600) / SAMPLE_RATE_HZ)
                        * 0.5).astype(np.float32),
    }
    offsets = np.array([direction for _, direction in MICROPHONE_LAYOUT]
                       ) * MICROPHONE_RADIUS_METERS
    for label, signal in sources.items():
        print(f"\nBEARING, noiseless, source at 3 m -- {label}")
        print("| true azimuth | measured | error | confidence |")
        print("|---|---|---|---|")
        errors = []
        for degrees in (0, 30, 45, 90, 135, 180, -45, -90):
            array.reset()
            radians = math.radians(degrees)
            source = np.array([3.0 * math.cos(radians),
                               3.0 * math.sin(radians), 0.0])
            ranges = np.linalg.norm(offsets - source, axis=1)
            channels = np.concatenate(
                [array.feed(signal[k:k + 320], ranges, 0.0)
                 for k in range(0, signal.size - 320, 320)], axis=1)
            azimuth, confidence = estimator.estimate(channels[:, -4800:])
            measured = math.degrees(azimuth) if azimuth is not None else float("nan")
            error = (measured - degrees + 180) % 360 - 180
            errors.append(abs(error))
            print(f"| {degrees:+4d}° | {measured:+7.1f}° | {error:+5.1f}° |"
                  f" {confidence:.2f} |")
        print(f"  mean |error| {np.mean(errors):.1f}°,"
              f" worst {np.max(errors):.1f}°")
    print("\n(no wind and no room: this table is the ARITHMETIC on its own."
          " The tables in test_hearing use real speech through the wind law.)")
