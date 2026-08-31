"""The sim's eyes-ears-follower brain on a REAL G1 -- same code, real sensors.

    python -m app.real.guide_walk --dry-run                # Mac, no hardware
    python -m app.real.guide_walk --iface eth0             # robot, print only
    python -m app.real.guide_walk --iface eth0 --arm       # robot, really walk

WHAT THIS IS. The demo harness's guide-following brain, unplugged from MuJoCo
and plugged into hardware. The BRAIN is imported from the sim modules verbatim
-- `detect_guide` (the HSV orange-backpack detector), `vector_command` (the
walk-toward-a-bearing steering law), `VoiceActivity` (webrtcvad),
`StopWord` (vosk, grammar ["stop", "[unk]"]), and the follow/wait hysteresis
constants -- so a behaviour tuned in the simulator is the behaviour on the
robot. Only the BODY changes:

    sim                                real
    ---------------------------------  -----------------------------------
    two MuJoCo cameras + SGBM          RealSense D435i (depth in hardware,
                                       so the stereo matcher is DELETED)
    synthesized 4-mic propagation      one real microphone (sounddevice)
    policy joint targets               LocoClient.Move(vx, vy, yaw) -- the
                                       onboard balance controller owns
                                       not-falling, this brain owns deciding

SAFETY, in order of importance:
  * `LocoClient.Move` expires after ~1 s on its own, and this loop re-issues
    it every tick -- so a crashed brain is a stopped robot within a second,
    by construction, not by cleanup code.
  * SIGINT/SIGTERM -> StopMove, then exit. Sending anything requires --arm;
    without it every command is printed and nothing moves.
  * Speeds are capped BELOW the sim's clamps (forward 0.4, lateral 0.2,
    yaw 0.6) because a first hardware run has no business at sim speed.

HONEST GAPS, stated rather than papered over:
  * ONE microphone -> no GCC-PHAT bearing. A voice heard while the guide is
    unseen therefore cannot be walked TOWARD; the robot instead turns slowly
    in place (SEARCH) until the detector acquires her, then vision navigates
    -- the sim's own division of labour, minus the ear compass. A 4-mic
    array restores the compass with the sim's own Bearing class.
  * The detector knows the answer's colour (the sim says the same of
    itself): the guide must wear the orange backpack.
  * No IMU tilt abort here: balance belongs to the onboard controller, and
    killing this process (the deadman above) is the stop.

FSM (mirrors app/harness/hearing.HearingBehaviour, single-mic edition):
    LISTENING  stand; a voice -> COMING (seen) or SEARCH (unseen)
    COMING     vector_command toward the detection; inside WAIT range -> WAIT
    SEARCH     rotate in place until seen (-> COMING) or timeout (-> LISTENING)
    WAIT       stand within arm's reach; she leaves range -> COMING
    STOPPED    the word `stop` at >= 0.90 confidence; ANY later voice resumes
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time

import numpy as np

from app.harness.guide import (
    FOLLOW_RANGE_METERS,
    FOLLOW_SPEED_METERS_PER_SECOND,
    WAIT_RANGE_METERS,
    detect_guide,
    vector_command,
)
from app.harness.hearing import (
    MICROPHONE_NOISE_GATE_RMS,
    SAMPLE_RATE_HZ,
    SEGMENT_END_SILENCE_SECONDS,
    SEGMENT_MAXIMUM_SECONDS,
    SEGMENT_MINIMUM_SECONDS,
    STOP_CONFIDENCE_THRESHOLD,
    StopWord,
    VoiceActivity,
)

CONTROL_HZ = 10.0                     # vision + decision rate (sim eyes: 10 Hz)
PRE_ROLL_SECONDS = 0.2                # audio glued ahead of VAD onset (sim value)
SEARCH_YAW_RADIANS_PER_SECOND = 0.4   # slow scan; full circle in ~16 s
SEARCH_TIMEOUT_SECONDS = 20.0
LOST_AFTER_SECONDS = 1.0              # sim's own detection-memory horizon
# Hardware caps UNDER the sim clamps -- first-run humility, see docstring.
MAXIMUM_FORWARD_METERS_PER_SECOND = 0.4
MAXIMUM_LATERAL_METERS_PER_SECOND = 0.2
MAXIMUM_YAW_RADIANS_PER_SECOND = 0.6
DEPTH_BOX_PERCENTILE = 50             # median depth inside the detection box


# ------------------------------------------------------------------ sensors
class RealSenseEyes:
    """D435i colour + aligned depth -> (rgb HxWx3 uint8, depth HxW metres).

    Depth comes from the camera's own stereo ASIC, so the sim's SGBM stage
    has no counterpart here -- it is not ported because it is not needed.
    Bearing convention matches the sim detector: +ve to the robot's LEFT.
    """

    def __init__(self, width=640, height=480, fps=30):
        import pyrealsense2 as rs
        self._rs = rs
        self.pipeline = rs.pipeline()
        configuration = rs.config()
        configuration.enable_stream(rs.stream.color, width, height,
                                    rs.format.rgb8, fps)
        configuration.enable_stream(rs.stream.depth, width, height,
                                    rs.format.z16, fps)
        profile = self.pipeline.start(configuration)
        self.align = rs.align(rs.stream.color)
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = stream.get_intrinsics()
        self.focal_pixels = float(intrinsics.fx)
        self.width = int(intrinsics.width)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        print(f"[real] D435i up: {self.width}px wide, focal"
              f" {self.focal_pixels:.1f}px, depth scale {self.depth_scale}",
              flush=True)

    def read(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(
            np.float32) * self.depth_scale
        return color, depth

    def bearing_radians(self, centre_column: float) -> float:
        """Pixel column -> body-frame bearing, +ve LEFT (the sim's sign)."""
        return float(math.atan2(self.width / 2.0 - centre_column,
                                self.focal_pixels))


class Microphone:
    """One real capsule -> gated 16 kHz float32 blocks, AGC-free.

    The same noise gate as the sim's MIC0 path (blocks under
    MICROPHONE_NOISE_GATE_RMS rms arrive as silence), because the problem it
    solves -- room noise opening voice segments -- is a real-room problem.
    """

    def __init__(self):
        import sounddevice
        self.buffer = []
        self.lock = threading.Lock()
        self.stream = sounddevice.InputStream(
            samplerate=SAMPLE_RATE_HZ, channels=1, dtype="float32",
            blocksize=int(SAMPLE_RATE_HZ * 0.032), callback=self._on_block)
        self.stream.start()
        print("[real] microphone up at 16 kHz, noise gate"
              f" {MICROPHONE_NOISE_GATE_RMS} rms", flush=True)

    def _on_block(self, indata, frame_count, time_info, status):
        samples = np.asarray(indata[:, 0], dtype=np.float32).copy()
        if float(np.sqrt(np.mean(samples ** 2))) < MICROPHONE_NOISE_GATE_RMS:
            samples = np.zeros_like(samples)
        with self.lock:
            self.buffer.append(samples)

    def take(self) -> np.ndarray:
        with self.lock:
            blocks, self.buffer = self.buffer, []
        return (np.concatenate(blocks) if blocks
                else np.zeros(0, dtype=np.float32))


# ---------------------------------------------------------------- the mouth
class LocoLegs:
    """LocoClient out. `--arm` decides whether Move is sent or printed.

    Move's ~1 s built-in expiry plus the per-tick re-issue is the deadman:
    see the module docstring. StopMove is sent on every transition to a
    standing mode AND from the signal handler.
    """

    def __init__(self, iface: str, armed: bool):
        self.armed = armed
        self.loco = None
        if iface is not None:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            ChannelFactoryInitialize(0, iface)
            self.loco = LocoClient()
            self.loco.SetTimeout(10.0)
            self.loco.Init()
            print(f"[real] LocoClient up on {iface}"
                  f" ({'ARMED' if armed else 'print-only'})", flush=True)

    def move(self, command) -> None:
        forward = float(np.clip(command[0], 0.0,
                                MAXIMUM_FORWARD_METERS_PER_SECOND))
        lateral = float(np.clip(command[1],
                                -MAXIMUM_LATERAL_METERS_PER_SECOND,
                                MAXIMUM_LATERAL_METERS_PER_SECOND))
        yaw_rate = float(np.clip(command[2],
                                 -MAXIMUM_YAW_RADIANS_PER_SECOND,
                                 MAXIMUM_YAW_RADIANS_PER_SECOND))
        if self.loco is not None and self.armed:
            self.loco.Move(forward, lateral, yaw_rate)

    def stop(self) -> None:
        if self.loco is not None and self.armed:
            self.loco.StopMove()


# ---------------------------------------------------------------- the brain
class GuideWalkBrain:
    """(detection, utterances) -> mode + [vx m/s, vy m/s, yaw rad/s].

    Pure decision logic, no hardware, no clock of its own -- `tick` is called
    at CONTROL_HZ with whatever the sensors saw. This is the class a unit
    test (and --dry-run) drives directly.
    """

    def __init__(self):
        self.mode = "LISTENING"
        self.seconds_since_seen = 1e9
        self.seconds_in_search = 0.0
        self.last_range = None
        self.last_bearing = 0.0

    def tick(self, seen: bool, range_meters, bearing_radians,
             utterance: str | None, dt: float) -> np.ndarray:
        """`utterance`: None, "voice", or "stop" (>= 0.90 confident)."""
        if seen:
            self.seconds_since_seen = 0.0
            self.last_range = float(range_meters)
            self.last_bearing = float(bearing_radians)
        else:
            self.seconds_since_seen += dt
        remembered = self.seconds_since_seen < LOST_AFTER_SECONDS

        if utterance == "stop":
            self.mode = "STOPPED"
        elif self.mode == "STOPPED":
            if utterance == "voice":
                self.mode = "LISTENING"
        elif utterance == "voice":
            self.mode = "COMING" if remembered else "SEARCH"
            self.seconds_in_search = 0.0

        if self.mode == "COMING":
            if not remembered:
                self.mode, self.seconds_in_search = "SEARCH", 0.0
            elif self.last_range is not None \
                    and self.last_range <= WAIT_RANGE_METERS:
                self.mode = "WAIT"
        elif self.mode == "WAIT":
            if remembered and self.last_range is not None \
                    and self.last_range > FOLLOW_RANGE_METERS:
                self.mode = "COMING"
        elif self.mode == "SEARCH":
            self.seconds_in_search += dt
            if remembered:
                self.mode = "COMING"
            elif self.seconds_in_search > SEARCH_TIMEOUT_SECONDS:
                self.mode = "LISTENING"

        if self.mode == "COMING":
            return vector_command(self.last_bearing,
                                  FOLLOW_SPEED_METERS_PER_SECOND)
        if self.mode == "SEARCH":
            return np.array([0.0, 0.0, SEARCH_YAW_RADIANS_PER_SECOND])
        return np.zeros(3)


class UtteranceMachine:
    """Gated PCM -> None | "voice" | "stop", once per closed utterance.

    The sim's segment law verbatim: VAD opens a segment, SEGMENT_END_SILENCE
    of quiet closes it, the whole utterance (plus PRE_ROLL) goes to vosk ONCE,
    and `stop` wins at >= STOP_CONFIDENCE_THRESHOLD.
    """

    def __init__(self, verbose=True):
        self.activity = VoiceActivity()
        self.words = StopWord(verbose=verbose)
        self.segment = None
        self.silence_seconds = 0.0
        self.tail = np.zeros(0, dtype=np.float32)

    def feed(self, samples: np.ndarray, dt: float):
        share = self.activity.feed(samples) if samples.size else 0.0
        voiced = share >= 0.30
        pre_roll = int(PRE_ROLL_SECONDS * SAMPLE_RATE_HZ)
        self.tail = np.concatenate((self.tail, samples))[-pre_roll:]
        if self.segment is None:
            if voiced:
                self.segment = [self.tail.copy()]
                self.silence_seconds = 0.0
            return None
        self.segment.append(samples)
        self.silence_seconds = 0.0 if voiced else self.silence_seconds + dt
        length = sum(part.size for part in self.segment) / SAMPLE_RATE_HZ
        if self.silence_seconds < SEGMENT_END_SILENCE_SECONDS \
                and length < SEGMENT_MAXIMUM_SECONDS:
            return None
        utterance = np.concatenate(self.segment)
        self.segment = None
        if utterance.size / SAMPLE_RATE_HZ < SEGMENT_MINIMUM_SECONDS:
            return None
        confidence = self.words.confidence(utterance)
        return ("stop" if confidence >= STOP_CONFIDENCE_THRESHOLD
                else "voice")


# ------------------------------------------------------------------- wiring
def perceive(rgb, depth, bearing_of_column):
    """One camera frame -> (seen, range_metres, bearing_radians)."""
    box, _mask = detect_guide(rgb)
    if box is None:
        return False, None, 0.0
    x0, y0, x1, y1 = box
    patch = depth[y0:y1 + 1, x0:x1 + 1]
    valid = patch[patch > 0.1]
    if valid.size == 0:
        return False, None, 0.0
    range_meters = float(np.percentile(valid, DEPTH_BOX_PERCENTILE))
    bearing = bearing_of_column((x0 + x1) / 2.0)
    return True, range_meters, bearing


def run(arguments) -> None:
    dt = 1.0 / CONTROL_HZ
    brain = GuideWalkBrain()
    ears = UtteranceMachine()

    if arguments.dry_run:
        _dry_run(brain, ears, dt)
        return

    eyes = RealSenseEyes()
    microphone = Microphone()
    legs = LocoLegs(arguments.iface, armed=arguments.arm)

    def halt(signal_number, frame):
        legs.stop()
        print("\n[real] StopMove sent; exiting.", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGINT, halt)
    signal.signal(signal.SIGTERM, halt)

    previous_mode = brain.mode
    while True:
        started = time.time()
        rgb, depth = eyes.read()
        seen, range_meters, bearing = perceive(rgb, depth,
                                               eyes.bearing_radians)
        utterance = ears.feed(microphone.take(), dt)
        command = brain.tick(seen, range_meters, bearing, utterance, dt)
        if brain.mode != previous_mode:
            print(f"[real] {previous_mode} -> {brain.mode}"
                  f" (seen={seen}, range={range_meters}, heard={utterance})",
                  flush=True)
            if brain.mode in ("WAIT", "STOPPED", "LISTENING"):
                legs.stop()
            previous_mode = brain.mode
        legs.move(command)
        if not arguments.arm:
            print(f"[real] {brain.mode:9s} cmd=[{command[0]:+.2f}"
                  f" {command[1]:+.2f} {command[2]:+.2f}]"
                  f" seen={seen} range={range_meters}", flush=True)
        time.sleep(max(0.0, dt - (time.time() - started)))


# ------------------------------------------------------------------ dry run
def _dry_run(brain, ears, dt) -> None:
    """No hardware: a synthetic guide walks the brain through every mode.

    Prints the command stream so the FSM and the reused sim brain can be
    eyeballed on any laptop. The 'voice' and 'stop' events are injected past
    the audio layer (vosk needs real speech); the AUDIO layer itself is
    exercised separately with silence and sub-gate noise, which must produce
    no utterance at all.
    """
    for label, samples in (("silence", np.zeros(1600, dtype=np.float32)),
                           ("sub-gate noise", (np.random.default_rng(0)
                            .normal(0.0, 0.005, 1600).astype(np.float32)))):
        gated = (np.zeros_like(samples)
                 if float(np.sqrt(np.mean(samples ** 2)))
                 < MICROPHONE_NOISE_GATE_RMS else samples)
        verdict = ears.feed(gated, 0.1)
        print(f"[dry] audio {label:14s} -> utterance {verdict}"
              f" (want None)", flush=True)

    def frame(seen, range_meters=None, bearing=0.0, heard=None):
        command = brain.tick(seen, range_meters, bearing, heard, dt)
        print(f"[dry] {brain.mode:9s} seen={str(seen):5s}"
              f" range={'None ' if range_meters is None else f'{range_meters:.2f}'}"
              f" heard={str(heard):5s} -> cmd=[{command[0]:+.2f}"
              f" {command[1]:+.2f} {command[2]:+.2f}]", flush=True)

    print("\n[dry] a voice with the guide unseen -> SEARCH (rotate):")
    frame(False, heard="voice")
    frame(False)
    print("[dry] the eyes acquire her at 4 m, 20 deg left -> COMING:")
    frame(True, 4.0, math.radians(20))
    frame(True, 3.0, math.radians(8))
    print("[dry] she is within arm's reach -> WAIT (stand):")
    frame(True, 1.2, 0.0)
    print("[dry] she walks off beyond 1.8 m -> COMING again:")
    frame(True, 2.5, math.radians(-5))
    print("[dry] a confident 'stop' -> STOPPED, even mid-approach:")
    frame(True, 2.5, 0.0, heard="stop")
    frame(True, 2.2, 0.0)
    print("[dry] any voice resumes -> LISTENING, then a call -> COMING:")
    frame(True, 2.2, 0.0, heard="voice")
    frame(True, 2.2, 0.0, heard="voice")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iface", default=None,
                        help="network interface to the robot, e.g. eth0")
    parser.add_argument("--arm", action="store_true",
                        help="actually send Move commands (default: print)")
    parser.add_argument("--dry-run", action="store_true",
                        help="no hardware; drive the brain through its modes")
    return parser


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    if not arguments.dry_run and arguments.iface is None:
        print("need --iface <robot interface> or --dry-run", file=sys.stderr)
        sys.exit(2)
    run(arguments)
