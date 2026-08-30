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
# 20 m/s gale on an UNWINDSHIELDED microphone sits about 10 dB below a shout at
# one metre. That fixes `WIND_NOISE_AT_REFERENCE` at 10^(-10/20) = 0.316 of
# `VOICE_AMPLITUDE_AT_ONE_METER`. A foam windshield would be worth 15-20 dB and
# this robot is not wearing one.
#
#     0 m/s -> 0.0005      6 m/s -> 0.029      12 m/s -> 0.115     20 m/s -> 0.316
#
# The floor is the electronics, not the weather: a real preamp is never silent.
WIND_NOISE_FLOOR = 0.0005
WIND_NOISE_AT_REFERENCE = 0.316
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
VOICE_ACTIVITY_WINDOW_SECONDS = 0.48      # 16 frames -> the reported share
VOICE_PRESENT_SHARE = 0.35        # above this share, "a human voice is present"
# A SEGMENT is one utterance: it opens when the VAD says voice and closes a
# short silence later. Both the word decision and the bearing are made ONCE per
# segment, on the whole utterance, which is why they are as good as they are --
# a 100 ms slice of "stop" is not a word and not a direction.
SEGMENT_END_SILENCE_SECONDS = 0.20
SEGMENT_MAXIMUM_SECONDS = 2.0
SEGMENT_MINIMUM_SECONDS = 0.12    # shorter than this is a click, not speech
# The bearing is measured on the LOUDEST window of the segment, because the
# quiet head and tail of an utterance are mostly the noise floor and GCC-PHAT
# would be cross-correlating wind with wind.
BEARING_WINDOW_SECONDS = 0.30
BEARING_INTERPOLATION_FACTOR = 8  # sub-sample resolution: 1/8 of 62.5 us
# How far below the strongest spectral bin the phase transform stops whitening.
# See `Bearing._pair` for the measurement that set it.
PHASE_TRANSFORM_FLOOR = 0.02
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
HEARING_MODES = ("IDLE", "COMING_BY_EYES", "COMING_BY_EARS", "STOPPED", "WAIT")
HEARING_MODE_CODES = {name: float(index)
                      for index, name in enumerate(HEARING_MODES)}
HEARD_CODES = {"none": 0.0, "voice": 1.0, "stop": 2.0}
# The walk toward a voice. Same forward speed the vision follower uses, so the
# hand-over from ears to eyes is not also a change of pace.
EAR_WALK_SPEED_METERS_PER_SECOND = 0.5
EAR_BEARING_GAIN_PER_RADIAN = 2.0
EAR_MAXIMUM_YAW_RATE_RADIANS_PER_SECOND = 1.0
EAR_BEARING_DEADBAND_RADIANS = math.radians(4.0)
# TURN FIRST, THEN WALK. Beyond this the robot pivots on the spot: walking while
# turning 120 degrees traces a long arc away from the person, and the whole
# point of the ear cue is that it is the only information there is.
EAR_TURN_IN_PLACE_BEYOND_RADIANS = math.radians(60.0)
# A bearing this uncertain is not a direction. Below it the cue is kept but the
# robot walks straight ahead rather than turning toward noise.
EAR_BEARING_MINIMUM_CONFIDENCE = 0.25
# How long an ear cue stays worth steering by with no new voice. She is not
# moving much, and re-shouting every second is not how people talk.
EAR_CUE_VALID_SECONDS = 20.0
NO_MEASUREMENT = -1.0             # hud.json cannot carry NaN; see guide.py


def wind_noise_amplitude(wind_speed_meters_per_second: float) -> float:
    """The noise floor at a given wind speed. -> amplitude, full scale = 1.0.

    The law is in this module's header and printed by `describe_wind_law()`.
    """
    speed = max(0.0, float(wind_speed_meters_per_second))
    return float(WIND_NOISE_FLOOR + WIND_NOISE_AT_REFERENCE
                 * (speed / WIND_NOISE_REFERENCE_MPS) ** 2)


def describe_wind_law() -> str:
    """One line, printed by every entry point that hears anything."""
    points = ", ".join(f"{speed:.0f} m/s -> {wind_noise_amplitude(speed):.4f}"
                       for speed in (0, 6, 12, 20))
    return (f"[hearing] wind noise = {WIND_NOISE_FLOOR}"
            f" + {WIND_NOISE_AT_REFERENCE} * (speed /"
            f" {WIND_NOISE_REFERENCE_MPS:.0f})^2  (quadratic: turbulent"
            f" pressure goes as the dynamic head; anchored at a 20 m/s gale"
            f" 10 dB under a shout at 1 m):  {points}"
            f"  -- independent draw per microphone")


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
        with self.lock:
            self.buffer = np.concatenate((self.buffer, samples))
            self.samples_received += int(samples.size)
            excess = self.buffer.size - self.capacity
            if excess > 0:
                self.buffer = self.buffer[excess:]
                self.samples_dropped += int(excess)
        return int(samples.size)

    def take(self, count: int) -> np.ndarray:
        """The next `count` samples, zero-padded if the page has not kept up."""
        count = int(count)
        with self.lock:
            available = min(count, self.buffer.size)
            chunk = self.buffer[:available].copy()
            self.buffer = self.buffer[available:]
        if available < count:
            self.samples_starved += count - available
            chunk = np.concatenate((chunk, np.zeros(count - available,
                                                    dtype=np.float32)))
        return chunk

    def clear(self) -> None:
        with self.lock:
            self.buffer = np.zeros(0, dtype=np.float32)


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
        from scipy.signal import lfilter
        filtered, state = lfilter([alpha], [1.0, -(1.0 - alpha)], signal,
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
        """Paul Kellet's three-row pink filter -- the page's own coefficients."""
        output = np.empty_like(white)
        row0, row1, row2 = (self.pink_rows[:, 0], self.pink_rows[:, 1],
                            self.pink_rows[:, 2])
        for index in range(white.shape[1]):
            sample = white[:, index]
            row0 = 0.99765 * row0 + sample * 0.0990460
            row1 = 0.96300 * row1 + sample * 0.2965164
            row2 = 0.57000 * row2 + sample * 1.0526913
            output[:, index] = (row0 + row1 + row2 + sample * 0.1848) * 0.22
        self.pink_rows[:, 0], self.pink_rows[:, 1], self.pink_rows[:, 2] = (
            row0, row1, row2)
        return output

    def _roll_off(self, signal):
        from scipy.signal import lfilter
        alpha = 1.0 - math.exp(-2.0 * math.pi * WIND_NOISE_LOW_PASS_HZ
                               / SAMPLE_RATE_HZ)
        output = np.empty_like(signal)
        for index in range(signal.shape[0]):
            filtered, state = lfilter([alpha], [1.0, -(1.0 - alpha)],
                                      signal[index],
                                      zi=[self.low_pass_state[index]])
            output[index] = filtered
            self.low_pass_state[index] = float(state[0])
        return output


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

    def feed(self, channel) -> None:
        """Append audio and consume whole 30 ms frames. Cheap; call at 10 Hz."""
        if not self.available:
            return
        started = time.time()
        self.pending = np.concatenate((self.pending,
                                       np.asarray(channel, dtype=np.float32)))
        while self.pending.size >= self.frame_samples:
            frame = self.pending[:self.frame_samples]
            self.pending = self.pending[self.frame_samples:]
            pcm = np.clip(frame * 32767.0, -32768, 32767).astype("<i2").tobytes()
            try:
                voiced = bool(self.detector.is_speech(pcm, SAMPLE_RATE_HZ))
            except Exception:
                voiced = False
            self.recent.append(voiced)
        if len(self.recent) > self.window_frames:
            self.recent = self.recent[-self.window_frames:]
        self.milliseconds = (time.time() - started) * 1000.0

    @property
    def probability(self) -> float:
        if not self.recent:
            return 0.0
        return float(np.mean(self.recent))

    @property
    def voice_present(self) -> bool:
        return self.probability >= VOICE_PRESENT_SHARE


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

    def __init__(self, verbose=True):
        self.available = False
        self.model = None
        self.milliseconds = 0.0
        try:
            import vosk
            vosk.SetLogLevel(-1)
            started = time.time()
            self.model = vosk.Model(lang="en-us")
            self.available = True
            if verbose:
                print(f"[hearing] vosk small en-us loaded in"
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
        confidence = float(np.clip((sharpness - 1.0) / 3.0, 0.0, 1.0))
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
        floor = float(np.sqrt(np.mean(window ** 2))) or 1e-12
        sharpness = float(window[peak] / floor)
        lag = float(np.clip(lags[peak], -self.maximum_lag_samples,
                            self.maximum_lag_samples))
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

        self._segment = []
        self._segment_channels = []
        self._silence_seconds = 0.0
        self._segment_seconds = 0.0
        self._detector_buffer = []      # channels awaiting a detector tick
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
        self._segment = []
        self._segment_channels = []
        self._silence_seconds = 0.0
        self._segment_seconds = 0.0
        self._detector_buffer = []

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
        voiced = self.voice_activity.voice_present
        seconds = block.shape[1] / SAMPLE_RATE_HZ

        if voiced:
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
        speech_seconds = self._segment_seconds - self._silence_seconds
        self._segment_seconds = 0.0
        self._silence_seconds = 0.0
        if speech_seconds < SEGMENT_MINIMUM_SECONDS:
            return
        self.segments_heard += 1
        self.new_segment = True
        self.stop_confidence = self.stop_word.confidence(monitor)
        self.heard = ("stop" if self.stop_confidence >= STOP_CONFIDENCE_THRESHOLD
                      else "voice")
        window = self._loudest_window(channels)
        azimuth, confidence = self.bearing_estimator.estimate(window)
        if azimuth is not None:
            self.bearing_radians = azimuth
            self.bearing_confidence = confidence
        else:
            self.bearing_confidence = 0.0

    @staticmethod
    def _loudest_window(channels):
        """The loudest `BEARING_WINDOW_SECONDS` of the utterance. -> (4, m).

        The head and tail of an utterance are mostly noise floor, and
        cross-correlating the noise floor with itself is how a bearing table
        comes out flat.
        """
        width = int(BEARING_WINDOW_SECONDS * SAMPLE_RATE_HZ)
        total = channels.shape[1]
        if total <= width:
            return channels
        energy = channels[0] ** 2
        cumulative = np.concatenate(([0.0], np.cumsum(energy)))
        sums = cumulative[width:] - cumulative[:-width]
        start = int(np.argmax(sums))
        return channels[:, start:start + width]

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
      * losing her mid-walk is the follower's SEARCH sweep, unchanged -- the
        ear layer simply hands its command straight through -- and a NEW shout
        re-cues by ear.

    WHY IDLE COMMANDS ZERO. With hearing on, the robot is waiting to be called;
    a robot that walks toward a person nobody called it to is a different demo.
    So `IDLE` is a standing robot, and the first utterance is what starts
    everything. With hearing OFF this class is never consulted and the guide
    follower's command goes through untouched.

    Inputs  : the `Ears` (its verdicts only), and the `GuideFollower`'s live
              state.
    Outputs : `mode` (one of `HEARING_MODES`) and `command()` -> (3,).
    """

    def __init__(self, walk_speed=EAR_WALK_SPEED_METERS_PER_SECOND):
        self.walk_speed = float(walk_speed)
        self.mode = "IDLE"
        self.stopped = False
        self.called = False
        self.cue_bearing_radians = None
        self.cue_confidence = 0.0
        self.cue_age_seconds = 0.0
        self._command = np.zeros(3)

    def reset(self) -> None:
        self.mode = "IDLE"
        self.stopped = False
        self.called = False
        self.cue_bearing_radians = None
        self.cue_confidence = 0.0
        self.cue_age_seconds = 0.0
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

    def update(self, ears: "Ears", follower, guide_command, dt_seconds: float):
        """One control tick. -> the (3,) command to fly."""
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
                    self.cue_confidence = float(ears.bearing_confidence)
                    self.cue_age_seconds = 0.0

        if self.stopped:
            self.mode = "STOPPED"
            self._command = np.zeros(3)
            return self._command

        if not self.called:
            self.mode = "IDLE"
            self._command = np.zeros(3)
            return self._command

        if self.eyes_have_her(follower):
            # VISION OWNS THE NAVIGATION from here. The follower's own command
            # is handed through byte for byte -- its hysteresis, its 1 m WAIT
            # band and its SEARCH sweep are all still exactly what they were.
            if follower.mode == "WAIT":
                self.mode = "WAIT"
                self.called = False       # arrived; the next shout re-calls
            else:
                self.mode = "COMING_BY_EYES"
            self._command = np.asarray(guide_command, dtype=float).copy()
            return self._command

        if (self.cue_bearing_radians is None
                or self.cue_age_seconds > EAR_CUE_VALID_SECONDS):
            # Called, but with no direction worth steering by. The follower's
            # SEARCH sweep is the right thing and it is already running.
            self.mode = "COMING_BY_EARS"
            self._command = np.asarray(guide_command, dtype=float).copy()
            return self._command

        self.mode = "COMING_BY_EARS"
        self._command = self._walk_toward(self.cue_bearing_radians)
        return self._command

    def _walk_toward(self, bearing_radians) -> np.ndarray:
        bearing = float(bearing_radians)
        if abs(bearing) < EAR_BEARING_DEADBAND_RADIANS:
            yaw_rate = 0.0
        else:
            yaw_rate = float(np.clip(
                EAR_BEARING_GAIN_PER_RADIAN * bearing,
                -EAR_MAXIMUM_YAW_RATE_RADIANS_PER_SECOND,
                EAR_MAXIMUM_YAW_RATE_RADIANS_PER_SECOND))
        forward = (0.0 if abs(bearing) > EAR_TURN_IN_PLACE_BEYOND_RADIANS
                   else self.walk_speed)
        return np.array([forward, 0.0, yaw_rate])

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
        return self.behaviour.update(self.ears, follower, guide_command,
                                     self.dt_seconds)

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
            "level_db": round(float(ears.level_db), 1),
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
