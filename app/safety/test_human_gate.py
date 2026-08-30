"""Gate unit tests on the raw G1 MJCF (the same `d435i` camera the team
model inherits; no jax needed). Run:
  python -m pytest app/safety/test_human_gate.py -q
"""
import os

import mujoco
import numpy as np
import pytest

from app.safety.human_gate import HumanGate, HumanWorld, VirtualFrustumDetector


@pytest.fixture(scope="module")
def sim():
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "assets", "robots", "mujoco", "g1_unitree.xml")
    model = mujoco.MjModel.from_xml_path(os.path.abspath(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def _yaw(quaternion_wxyz):
    w, x, y, z = quaternion_wxyz
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def test_human_ahead_blocks_up(sim):
    model, data = sim
    world = HumanWorld()
    world.spawn_ahead_of(data.qpos[0:3], _yaw(data.qpos[3:7]), 1.2)
    gate = HumanGate(VirtualFrustumDetector(model, world), clear_after_seconds=1.0)
    gate.update(data, 0.0)
    assert not gate.allow_up()
    assert gate.mask(np.array([0.5, 0.0, 0.1])).tolist() == [0.0, 0.0, 0.1]
    assert gate.mask(np.array([-0.5, 0.0, 0.0]))[0] == -0.5   # down still allowed


def test_human_behind_or_far_is_clear(sim):
    model, data = sim
    world = HumanWorld()
    yaw = _yaw(data.qpos[3:7])
    world.spawn_ahead_of(data.qpos[0:3], yaw, -1.2)   # behind
    world.spawn_ahead_of(data.qpos[0:3], yaw, 6.0)    # too far
    gate = HumanGate(VirtualFrustumDetector(model, world))
    gate.update(data, 0.0)
    assert gate.allow_up()


def test_hysteresis(sim):
    model, data = sim
    world = HumanWorld()
    world.spawn_ahead_of(data.qpos[0:3], _yaw(data.qpos[3:7]), 1.0)
    gate = HumanGate(VirtualFrustumDetector(model, world), clear_after_seconds=1.0)
    gate.update(data, 0.0)
    world.clear()
    gate.update(data, 0.5)
    assert not gate.allow_up()       # still inside the hold
    gate.update(data, 1.1)
    assert gate.allow_up()


def test_torso_fallback_matches_real_camera(sim):
    """A model without the d435i must place the virtual camera exactly where
    the XML camera is."""
    model, data = sim
    real = VirtualFrustumDetector(model, HumanWorld())
    fallback = VirtualFrustumDetector(model, HumanWorld(), camera_name="nope")
    p_real, r_real = real.camera_pose(data)
    p_fake, r_fake = fallback.camera_pose(data)
    assert np.allclose(p_real, p_fake, atol=1e-6)
    assert np.allclose(r_real, r_fake, atol=1e-6)
