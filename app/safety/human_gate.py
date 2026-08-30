"""Human-detection gate: the robot may not climb UP while a human is in front.

This is a DETERMINISTIC runtime layer, not a policy. The PPO policy stays a pure
"climb given a command" controller; the gate sits between the joystick and the
policy and masks the forward command. A safety rule that must be auditable
cannot live inside a probabilistic network.

Three pieces, kept apart so the sim oracle and the real detector are swappable:

    HumanWorld        the humans we spawn in simulation (positions + a capsule
                      drawn into the render scene). The compiled MjModel comes
                      from THEIR env and is never edited, so humans are virtual:
                      no physics, no collision, only a position and a picture.
    HumanDetector     `detect(data) -> Detection`. Two implementations:
                        VirtualFrustumDetector  sim oracle: projects each human
                                                into the `d435i` camera frustum
                                                (same fov as the real RealSense)
                        (later) RealSenseYoloDetector on the Jetson, same output
    HumanGate         the state machine. Seen -> blocked at once. Cleared only
                      after `clear_after_seconds` of no detections (hysteresis,
                      so a flickering detector cannot re-arm the climb).

Camera convention (MuJoCo): a camera looks along its local -z, +y is up, +x is
right. `data.cam_xpos` / `data.cam_xmat` give the world pose after mj_forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import mujoco
import numpy as np

CAMERA_NAME = "d435i"           # assets/robots/mujoco/g1_unitree.xml:289
# The same camera, as an offset on torso_link, for models that lack it (the
# TEAM model is compiled from mujoco_playground's G1 XML, which has no d435i).
# pos / xyaxes copied verbatim from g1_unitree.xml:289; fovy 58 deg = RealSense.
CAMERA_PARENT_BODY = "torso_link"
CAMERA_OFFSET_POSITION = np.array([0.0789635, 0.0, 0.386])
CAMERA_OFFSET_ROTATION = np.array([[0.0, 0.0, -1.0],    # columns = cam x, y, z
                                   [-1.0, 0.0, 0.0],    # in torso frame
                                   [0.0, 1.0, 0.0]])
CAMERA_FOVY_DEGREES = 58.0
DEFAULT_MAX_RANGE_METERS = 2.0  # "human in front" means closer than this
HUMAN_HEIGHT_METERS = 1.7
HUMAN_RADIUS_METERS = 0.2


@dataclass
class Detection:
    seen: bool
    distance_meters: float | None = None   # nearest human seen, else None
    bearing_radians: float | None = None   # + left / - right of optical axis
    count: int = 0


@dataclass
class Human:
    position_world: np.ndarray             # (3,) feet on the ground
    name: str = "human"


@dataclass
class HumanWorld:
    """The virtual humans. Positions are world frame, z = ground contact."""
    humans: list = field(default_factory=list)

    def spawn(self, position_world, name=None) -> Human:
        human = Human(np.asarray(position_world, dtype=float),
                      name or f"human_{len(self.humans)}")
        self.humans.append(human)
        return human

    def spawn_ahead_of(self, root_position_world, root_yaw_radians,
                       distance_meters, lateral_meters=0.0, name=None) -> Human:
        """Spawn `distance_meters` ahead of the robot along its yaw."""
        forward = np.array([math.cos(root_yaw_radians), math.sin(root_yaw_radians), 0.0])
        left = np.array([-forward[1], forward[0], 0.0])
        position = (np.asarray(root_position_world, dtype=float)
                    + distance_meters * forward + lateral_meters * left)
        position[2] = 0.0
        return self.spawn(position, name)

    def clear(self) -> None:
        self.humans.clear()

    def centers_world(self) -> np.ndarray:
        """(n, 3) torso centres -- what a detector is looking for."""
        if not self.humans:
            return np.zeros((0, 3))
        centers = np.stack([h.position_world for h in self.humans])
        centers[:, 2] += HUMAN_HEIGHT_METERS / 2.0
        return centers

    def draw(self, scene: mujoco.MjvScene, rgba=(0.9, 0.2, 0.2, 0.8)) -> None:
        """Add one capsule per human to a renderer scene (call after
        `renderer.update_scene`, before `renderer.render`)."""
        for center in self.centers_world():
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            half_length = HUMAN_HEIGHT_METERS / 2.0 - HUMAN_RADIUS_METERS
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.array([HUMAN_RADIUS_METERS, half_length, 0.0]),
                center, np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
            scene.ngeom += 1


class HumanDetector:
    def detect(self, data: mujoco.MjData) -> Detection:
        raise NotImplementedError


class VirtualFrustumDetector(HumanDetector):
    """Sim oracle. A human is 'seen' when its torso centre lies inside the
    camera's field of view and within `max_range_meters`. No occlusion test:
    this is deliberately optimistic about what the camera can see, which is the
    conservative direction for a gate (more blocking, never less)."""

    def __init__(self, model: mujoco.MjModel, world: HumanWorld,
                 max_range_meters=DEFAULT_MAX_RANGE_METERS,
                 aspect=640.0 / 480.0, camera_name=CAMERA_NAME):
        self.model = model
        self.world = world
        self.max_range_meters = float(max_range_meters)
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id >= 0:
            self.parent_body_id = -1
            fovy = math.radians(float(model.cam_fovy[self.camera_id]))
        else:
            self.parent_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, CAMERA_PARENT_BODY)
            if self.parent_body_id < 0:
                raise ValueError(f"neither camera {camera_name!r} nor body"
                                 f" {CAMERA_PARENT_BODY!r} in model")
            fovy = math.radians(CAMERA_FOVY_DEGREES)
        self.half_fovy = fovy / 2.0
        self.half_fovx = math.atan(math.tan(self.half_fovy) * aspect)

    def camera_pose(self, data: mujoco.MjData):
        """(position (3,), rotation (3,3) with columns = camera axes), world."""
        if self.camera_id >= 0:
            return (np.asarray(data.cam_xpos[self.camera_id]),
                    np.asarray(data.cam_xmat[self.camera_id]).reshape(3, 3))
        body_pos = np.asarray(data.xpos[self.parent_body_id])
        body_rot = np.asarray(data.xmat[self.parent_body_id]).reshape(3, 3)
        return (body_pos + body_rot @ CAMERA_OFFSET_POSITION,
                body_rot @ CAMERA_OFFSET_ROTATION)

    def detect(self, data: mujoco.MjData) -> Detection:
        centers = self.world.centers_world()
        if len(centers) == 0:
            return Detection(seen=False)
        cam_pos, cam_rot = self.camera_pose(data)
        local = (centers - cam_pos) @ cam_rot            # world -> camera frame
        depth = -local[:, 2]                             # camera looks along -z
        in_front = depth > 1e-3
        bearing = np.arctan2(local[:, 0], np.maximum(depth, 1e-3))   # +x right
        elevation = np.arctan2(local[:, 1], np.maximum(depth, 1e-3))
        distance = np.linalg.norm(local, axis=1)
        visible = (in_front & (np.abs(bearing) <= self.half_fovx)
                   & (np.abs(elevation) <= self.half_fovy)
                   & (distance <= self.max_range_meters))
        if not visible.any():
            return Detection(seen=False)
        nearest = int(np.argmin(np.where(visible, distance, np.inf)))
        return Detection(seen=True,
                         distance_meters=float(distance[nearest]),
                         bearing_radians=float(-bearing[nearest]),
                         count=int(visible.sum()))


class HumanGate:
    """Blocks the UP command while a human is in front, with hysteresis."""

    def __init__(self, detector: HumanDetector, clear_after_seconds=1.0):
        self.detector = detector
        self.clear_after_seconds = float(clear_after_seconds)
        self.blocked = False
        self.last_seen_seconds = None
        self.last_detection = Detection(seen=False)

    def update(self, data: mujoco.MjData, time_seconds: float) -> Detection:
        detection = self.detector.detect(data)
        self.last_detection = detection
        if detection.seen:
            self.blocked = True
            self.last_seen_seconds = time_seconds
        elif self.blocked and self.last_seen_seconds is not None and (
                time_seconds - self.last_seen_seconds >= self.clear_after_seconds):
            self.blocked = False
        return detection

    def allow_up(self) -> bool:
        return not self.blocked

    def mask(self, command: np.ndarray) -> np.ndarray:
        """Their (3,) [lin_vel_x, lin_vel_y, ang_vel_yaw]. Forward = up the
        rope, so a blocked gate clamps lin_vel_x to <= 0. Down and yaw pass."""
        if self.allow_up():
            return command
        masked = np.array(command, dtype=float, copy=True)
        masked[0] = min(masked[0], 0.0)
        return masked

    def state(self) -> dict:
        d = self.last_detection
        return {"human_clear": self.allow_up(), "human_seen": d.seen,
                "human_count": d.count,
                "human_distance_meters": d.distance_meters,
                "human_bearing_degrees": (None if d.bearing_radians is None
                                          else math.degrees(d.bearing_radians))}
