"""Offline correctness tests for the onboard deployment package.

These pin the deployment code to the training-time definition. Two bugs were
already caught here and both were silent:
  - gravity used R's third column instead of its third row (a transpose slip
    that is still unit-norm and yaw-invariant, so hand-checks pass);
  - the fused swish aliased its input buffer and returned all ones.

Neither would have raised on the robot. They would have looked like a policy
that "just doesn't transfer".

The MuJoCo cross-check needs the playground G1 XML with menagerie meshes
stripped; it skips if that fixture is absent. Everything else runs anywhere.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy import constants as C
from deploy.observation import ObservationBuilder, gravity_from_quaternion
from deploy.policy import Policy

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPZ = os.path.join(_ROOT, "rl", "policies", "mels_g1_joystick.npz")
FIXTURE = os.environ.get("G1_NOMESH_SCENE", "")


def test_obs_slices_partition_the_vector():
    cov = np.zeros(C.OBS_DIM, dtype=int)
    for s in (C.SLICE_LINVEL, C.SLICE_GYRO, C.SLICE_GRAVITY, C.SLICE_COMMAND,
              C.SLICE_JOINT_POS, C.SLICE_JOINT_VEL, C.SLICE_LAST_ACT,
              C.SLICE_PHASE):
        cov[s] += 1
    assert (cov == 1).all(), "obs slices must tile [0, 103) exactly once"


def test_gravity_is_yaw_invariant_and_unit_norm():
    for yaw in np.linspace(-np.pi, np.pi, 32):
        q = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        np.testing.assert_allclose(gravity_from_quaternion(q),
                                   [0, 0, -1], atol=1e-6)
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        assert abs(np.linalg.norm(gravity_from_quaternion(q)) - 1.0) < 1e-6


@pytest.mark.skipif(not FIXTURE or not os.path.exists(FIXTURE),
                    reason="set G1_NOMESH_SCENE to the mesh-stripped playground scene")
def test_gravity_matches_mujoco_pelvis_imu_site():
    """Ground truth: playground's `site_xmat[imu_in_pelvis].T @ [0,0,-1]`."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(FIXTURE)
    d = mujoco.MjData(m)
    sid = m.site("imu_in_pelvis").id
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        d.qpos[:] = 0; d.qpos[2] = 0.755; d.qpos[3:7] = q
        d.qpos[7:] = rng.uniform(-0.5, 0.5, m.nq - 7)
        mujoco.mj_forward(m, d)
        ref = d.site_xmat[sid].reshape(3, 3).T @ np.array([0, 0, -1.0])
        worst = max(worst, float(np.abs(ref - gravity_from_quaternion(q)).max()))
    assert worst < 1e-6, f"gravity mismatch vs MuJoCo: {worst:.2e}"


@pytest.mark.skipif(not FIXTURE or not os.path.exists(FIXTURE),
                    reason="set G1_NOMESH_SCENE to the mesh-stripped playground scene")
def test_joint_order_and_default_pose_match_the_model():
    import mujoco
    m = mujoco.MjModel.from_xml_path(FIXTURE)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(m.nu)]
    assert [n.replace("_joint", "") for n in names] == list(C.JOINT_NAMES)
    np.testing.assert_allclose(m.keyframe("knees_bent").ctrl,
                               C.DEFAULT_POSE, atol=1e-6)


@pytest.mark.skipif(not os.path.exists(NPZ), reason="policy npz not present")
def test_optimized_forward_matches_naive_reference():
    z = np.load(NPZ)

    def reference(obs):
        x = (obs - z["obs_mean"]) / z["obs_std"]
        sw = lambda v: v / (1.0 + np.exp(-v))
        for i in range(3):
            x = sw(x @ z[f"hidden_{i}_kernel"] + z[f"hidden_{i}_bias"])
        return (x @ z["hidden_3_kernel"] + z["hidden_3_bias"])[:C.ACTION_DIM]

    p = Policy(NPZ)
    rng = np.random.default_rng(0)
    with np.errstate(over="ignore"):   # swish saturates on synthetic obs
        for _ in range(500):
            o = rng.normal(0, 1, C.OBS_DIM).astype(np.float32)
            np.testing.assert_allclose(p(o), reference(o), atol=1e-5)


@pytest.mark.skipif(not os.path.exists(NPZ), reason="policy npz not present")
def test_golden_standing_action():
    """Pin policy output so a refactor cannot silently move it."""
    p = Policy(NPZ)
    b = ObservationBuilder()
    obs = b.build(C.DEFAULT_POSE, np.zeros(29), np.zeros(3), [1, 0, 0, 0],
                  [0, 0, 0])
    dev = np.abs(p.motor_targets(p(obs)) - C.DEFAULT_POSE).max()
    assert abs(dev - 0.367) < 0.01, f"standing action drifted: {dev:.4f}"


@pytest.mark.skipif(not os.path.exists(NPZ), reason="policy npz not present")
def test_policy_does_not_allocate_in_the_hot_loop():
    p = Policy(NPZ)
    obs = np.zeros(C.OBS_DIM, dtype=np.float32)
    a1 = p(obs)
    assert a1.dtype == np.float32 and a1.shape == (C.ACTION_DIM,)
    # The return is a view into the reused output buffer: a fresh ndarray
    # object each call, but no data allocation. Check the buffer, not identity.
    a2 = p(obs)
    assert np.shares_memory(a1, a2), "output buffer should be reused"
    for name, buf in [("x", p._x), *[(f"h{i}", h) for i, h in enumerate(p._h)]]:
        assert buf.dtype == np.float32, f"scratch {name} must stay fp32"


def test_command_is_clamped_to_trained_range():
    b = ObservationBuilder()
    obs = b.build(C.DEFAULT_POSE, np.zeros(29), np.zeros(3), [1, 0, 0, 0],
                  [99.0, -99.0, 99.0])
    np.testing.assert_allclose(obs[C.SLICE_COMMAND], [1.0, -0.5, 1.0])


def test_gait_clock_advances_and_wraps():
    b = ObservationBuilder()
    np.testing.assert_allclose(b.phase, C.PHASE_INIT)
    for _ in range(1000):
        b.advance_phase()
        assert np.all(b.phase > -np.pi - 1e-6) and np.all(b.phase <= np.pi + 1e-6)


def test_builder_rejects_wrong_joint_count():
    b = ObservationBuilder()
    with pytest.raises(ValueError):
        b.build(np.zeros(23), np.zeros(23), np.zeros(3), [1, 0, 0, 0], [0, 0, 0])


# --- observation views: one canonical obs feeding several policy variants ----

def _synth_npz(obs_dim, tmp_path, seed=0):
    """A structurally valid checkpoint with the given input width."""
    rng = np.random.default_rng(seed)
    dims = [obs_dim, 512, 256, 128, 2 * C.ACTION_DIM]
    d = {f"hidden_{i}_kernel": rng.normal(0, 0.05, (dims[i], dims[i + 1])
                                          ).astype(np.float32) for i in range(4)}
    d.update({f"hidden_{i}_bias": np.zeros(dims[i + 1], np.float32)
              for i in range(4)})
    d["obs_mean"] = np.zeros(obs_dim, np.float32)
    d["obs_std"] = np.ones(obs_dim, np.float32)
    path = str(tmp_path / f"synth_{obs_dim}.npz")
    np.savez(path, **d)
    return path


def test_view_inferred_from_checkpoint_width(tmp_path):
    p = Policy(_synth_npz(100, tmp_path))
    assert p.obs_dim == 100 and not p.uses_linvel
    p103 = Policy(_synth_npz(103, tmp_path))
    assert p103.obs_dim == 103 and p103.uses_linvel


def test_unknown_obs_width_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no known observation view"):
        Policy(_synth_npz(97, tmp_path))


def test_no_linvel_policy_is_structurally_blind_to_linvel(tmp_path):
    """A 100-dim policy must be *unable* to see dims 0:3, not merely trained
    to ignore them. This is what makes side-by-side comparison meaningful."""
    p = Policy(_synth_npz(100, tmp_path))
    b = ObservationBuilder()
    obs = b.build(C.DEFAULT_POSE, np.zeros(29), np.zeros(3), [1, 0, 0, 0],
                  [1, 0, 0])
    a = p(obs).copy()
    for probe in ([5.0, -3.0, 2.0], [-1.0, 1.0, -1.0], [0.6, -0.2, 0.1]):
        obs[C.SLICE_LINVEL] = probe
        np.testing.assert_array_equal(p(obs), a)


@pytest.mark.skipif(not os.path.exists(NPZ), reason="policy npz not present")
def test_both_policies_run_from_one_canonical_observation(tmp_path):
    full, novel = Policy(NPZ), Policy(_synth_npz(100, tmp_path))
    b = ObservationBuilder()
    obs = b.build(C.DEFAULT_POSE, np.zeros(29), np.zeros(3), [1, 0, 0, 0],
                  [1, 0, 0])
    a_full, a_novel = full(obs).copy(), novel(obs).copy()
    assert a_full.shape == a_novel.shape == (C.ACTION_DIM,)
    assert not np.allclose(a_full, a_novel)


@pytest.mark.skipif(not os.path.exists(NPZ), reason="policy npz not present")
def test_action_buffer_is_invalidated_by_the_next_call():
    """Documents the reuse hazard: two policies in one loop MUST copy."""
    p = Policy(NPZ)
    b = ObservationBuilder()
    o1 = b.build(C.DEFAULT_POSE, np.zeros(29), np.zeros(3), [1, 0, 0, 0],
                 [1, 0, 0]).copy()
    o2 = o1.copy()
    o2[C.SLICE_COMMAND] = [-1.0, 0.0, 0.0]
    a1_view = p(o1)          # not copied -- deliberately
    a1_copy = a1_view.copy()
    a2 = p(o2)
    assert np.shares_memory(a1_view, a2)
    assert not np.allclose(a1_copy, a2), "the two commands should differ"
    np.testing.assert_array_equal(a1_view, a2)  # the view followed the buffer
