"""Acceptance tests for the merged G1 + terrain + rope scene.

Several of these lock down bugs that were live and silent: each one produced a
model that compiled, ran, and looked plausible while being wrong.
"""
import mujoco
import numpy as np
import pytest

from rl.environment import ascender as A
from rl.environment import climb_scene as CS
from rl.environment import terrain as T


# -- terrain ---------------------------------------------------------------

def test_patches_share_a_grid():
    """All patches must share a shape, or MJX cannot batch them."""
    shapes = {n: T.load_patch(n).shape
              for names in T.list_patches().values() for n in names}
    assert len(set(shapes.values())) == 1, shapes


@pytest.mark.parametrize("name", ["A", "B", "C", "D"])
def test_deplane_recovers_the_generator(name):
    """Patches are built as plane + noise, so de-planing must return the noise."""
    tr = T.load_patch(name)
    assert 0.05 < tr.rough.std() < 0.2       # ~0.12 m rms by construction
    assert abs(tr.rough.mean()) < 1e-6       # mean-zero
    assert 20.0 < tr.slope_deg < 55.0


def test_roughness_matches_build_patch_set_recipe():
    a = T.synth_roughness(64, 96, 0.05, 0.12, 7)
    b = T.synth_roughness(64, 96, 0.05, 0.12, 7)
    assert np.allclose(a, b)                        # seeded, reproducible
    assert T.synth_roughness(64, 96, 0.05, 0.12, 8).std() == pytest.approx(0.12, rel=1e-6)
    assert not np.allclose(a, T.synth_roughness(64, 96, 0.05, 0.12, 8))


@pytest.mark.parametrize(
    "tr,tol",
    [
        (T.make_terrain(25, 0.12, 1), 0.02),
        (T.make_terrain(50, 0.12, 2), 0.02),
        (T.load_patch("B"), 0.02),
        (T.load_patch_baked("B"), 0.05),
    ],
)
def test_surface_z_matches_the_compiled_heightfield(tr, tol):
    """terrain.surface_z must agree with what MuJoCo actually collides against.

    The rope is hung relative to surface_z, so an error here puts the line
    underground. Two earlier bugs lived exactly here: a sign flip on the baked
    geom offset, and a dropped h*sin(slope) term in the separable inverse.
    """
    sc = CS.build_scene(tr)
    mask = np.zeros(6, np.uint8)
    mask[CS.GROUP_TERRAIN] = 1
    ext = tr.world_extent_x - 0.5
    for x in np.linspace(-ext, ext, 13):
        for y in (-3.0, 0.0, 3.0):
            top = float(tr.surface_z(x, y)) + 30.0
            gid = np.zeros(1, np.int32)
            dist = mujoco.mj_ray(
                sc.model, sc.data, np.array([x, y, top]),
                np.array([0.0, 0.0, -1.0]), mask, 1, -1, gid,
            )
            name = mujoco.mj_id2name(sc.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0]))
            if name != "floor":
                continue
            assert abs(top - dist - float(tr.surface_z(x, y))) < tol


def test_distinct_terrains_do_not_share_a_heightfield_file():
    """MuJoCo caches assets by path; same-named grids silently collided."""
    sep, baked = T.load_patch("B"), T.load_patch_baked("B")
    a, b = CS.build_scene(sep), CS.build_scene(baked)
    assert not np.allclose(a.model.hfield_data, b.model.hfield_data)


# -- rope and ascender -----------------------------------------------------

def test_route_projection_round_trips():
    r = A.drape_route(T.make_terrain(38, 0.12, 3))
    for s in np.linspace(0, r.length, 40):
        assert r.project_arclen(r.point_at(s))[0] == pytest.approx(s, abs=1e-9)


def test_ratchet_never_gives_ground_back():
    r = A.drape_route(T.make_terrain(38, 0.12, 3))
    asc = A.MocapAscender(r, s0=0.0)
    asc.update(r.point_at(5.0))
    asc.update(r.point_at(1.0))          # a slip
    assert asc.s == pytest.approx(5.0)
    asc.update(r.point_at(6.0))
    assert asc.s == pytest.approx(6.0)


def test_ratchet_can_be_disabled_for_ablation():
    r = A.drape_route(T.make_terrain(38, 0.12, 3))
    asc = A.MocapAscender(r, s0=0.0, ratchet=False)
    asc.update(r.point_at(5.0))
    asc.update(r.point_at(1.0))
    assert asc.s == pytest.approx(1.0)


# -- merged scene ----------------------------------------------------------

def test_mocap_carrier_adds_no_degrees_of_freedom():
    """The whole point of a mocap carrier over a slide joint.

    A slide joint appends a 30th coordinate that every qpos[7:] slice upstream
    picks up as a phantom joint; nq must stay at the robot's own 36.
    """
    sc = CS.build_scene(T.load_patch("B"))
    assert (sc.model.nq, sc.model.nv, sc.model.nu) == (36, 35, 29)
    assert sc.model.nmocap == 1


def test_hand_starts_exactly_on_the_rope():
    """A non-zero initial equality error is yanked out by the stiff grip."""
    assert CS.build_scene(T.load_patch("B")).hand_rope_distance() < 1e-6


def test_reset_is_idempotent_and_uses_fresh_kinematics():
    """reset() must forward-evaluate before placing the carrier.

    mj_resetDataKeyframe writes qpos but leaves site_xpos stale; reading the
    palm too early placed the carrier metres away and detonated the solver.
    """
    sc = CS.build_scene(T.load_patch("B"))
    for _ in range(3):
        sc.reset()
        assert sc.hand_rope_distance() < 1e-6
        assert np.linalg.norm(
            sc.data.mocap_pos[sc.carrier_mocap_id] - sc.palm_xyz
        ) < 1e-9


def test_foot_contact_pairs_track_the_ice_setting():
    """The G1 XML pins foot-floor friction via an explicit <pair>.

    Setting geom_friction alone leaves the real coefficient at 0.6 and the
    icyness axis does nothing whatsoever.
    """
    for mu in (0.9, 0.05):
        sc = CS.build_scene(T.load_patch("B"), friction=CS.FrictionParams.from_scalar(mu))
        m = sc.model
        for name in ("left_foot_floor", "right_foot_floor"):
            pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, name)
            assert pid >= 0
            assert m.pair_friction[pid][0] == pytest.approx(mu)


def test_terrain_geom_keeps_the_floor_name():
    """The foot contact sensors resolve their pair by the name `floor`."""
    m = CS.build_scene(T.load_patch("B")).model
    gid = m.geom("floor").id
    assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_floor_found") >= 0


def test_rope_is_visible_by_default():
    """Group 3 is hidden by MuJoCo's default view mask; the rope must show."""
    m = CS.build_scene(T.load_patch("B")).model
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    gid = m.geom("ropeseg0").id
    assert opt.geomgroup[m.geom_group[gid]] == 1


def test_rope_clears_the_terrain_along_its_whole_length():
    sc = CS.build_scene(T.load_patch("B"))
    for s in np.linspace(0, sc.route.length, 60):
        p = sc.route.point_at(s)
        assert p[2] - float(sc.terrain.surface_z(p[0], p[1])) > 0.1


def test_rope_arrests_a_fall_that_would_otherwise_be_terminal():
    def sag(grip):
        sc = CS.build_scene(T.load_patch("B"))
        m, d = sc.model, sc.data
        if not grip:
            m.eq_active0[m.equality("ascender_grip").id] = 0
        sc.reset()
        ctrl = m.key(sc.key_id).ctrl.copy()
        z0 = float(d.qpos[2])
        for _ in range(int(4.0 / m.opt.timestep)):
            d.ctrl[:] = ctrl
            sc.step()
        assert np.isfinite(d.qpos).all()
        return float(d.qpos[2]) - z0, sc

    roped, sc = sag(True)
    unroped, _ = sag(False)
    assert sc.hand_rope_distance() < 1e-3      # still gripping
    assert abs(roped) < 0.6 * abs(unroped)     # and materially higher up


def test_export_round_trips_to_a_standalone_model(tmp_path):
    sc = CS.build_scene(T.load_patch("B"))
    path = sc.export(str(tmp_path / "scene.xml"))
    m = mujoco.MjModel.from_xml_path(path)
    assert (m.nq, m.nv, m.nu) == (sc.model.nq, sc.model.nv, sc.model.nu)
    assert m.nhfield == 1 and m.neq == 1 and m.nmocap == 1
    assert np.allclose(m.hfield_data, sc.model.hfield_data)


# -- walking policy --------------------------------------------------------

def test_policy_observation_is_the_trained_layout():
    """103 dims in playground's order, or the policy sees an unseen input."""
    from rl.environment import walk_policy as WP

    sc = CS.build_scene(T.make_terrain(0, 0.0, 1))
    ctl = WP.WalkController(sc.model, command=(0.3, 0.0, 0.0))
    obs = ctl.observe(sc.data)
    assert obs.shape == (WP.OBS_DIM,)
    assert np.allclose(obs[9:12], [0.3, 0.0, 0.0])          # command block
    assert np.allclose(obs[70:99], 0.0)                     # last_act, at reset
    # phase is [cos p0, cos p1, sin p0, sin p1] with p = (0, pi)
    assert np.allclose(obs[99:103], [1.0, -1.0, 0.0, 0.0], atol=1e-9)
    # upright robot -> gravity in the pelvis frame points down
    assert obs[8] < -0.9


def test_policy_is_deterministic_and_finite_on_extreme_input():
    from rl.environment import walk_policy as WP

    p = WP.WalkPolicy()
    o = np.full(WP.OBS_DIM, 100.0)
    assert np.allclose(p(o), p(o))
    assert np.isfinite(p(o)).all()      # the naive swish overflows here


def test_policy_stands_on_flat_ground_but_keyframe_hold_does_not():
    """The reference behaviour: a fall with no policy is not a model fault."""
    from rl.environment import walk_policy as WP

    def run(use_policy):
        sc = CS.build_scene(T.make_terrain(0, 0.0, 1))
        m, d = sc.model, sc.data
        m.eq_active0[m.equality("ascender_grip").id] = 0
        sc.reset()
        ctl = WP.WalkController(m, command=(0.0, 0.0, 0.0)) if use_policy else None
        kctrl = m.key(sc.key_id).ctrl.copy()
        z0 = float(d.qpos[2])
        for _ in range(int(6.0 / m.opt.timestep)):
            if ctl:
                ctl.substep(d)
            else:
                d.ctrl[:] = kctrl
            sc.step()
        upright = float(d.site_xmat[m.site("imu_in_pelvis").id].reshape(3, 3)[2, 2])
        return float(d.qpos[2]) - z0, upright

    dz_pol, up_pol = run(True)
    dz_key, up_key = run(False)
    assert abs(dz_pol) < 0.15 and up_pol > 0.9      # still standing
    assert up_key < 0.5                             # toppled


# -- project gear from g1_unitree.usd --------------------------------------

def _gear_available():
    from rl.tools import usd_gear

    return bool(usd_gear.load_manifest()[1])


@pytest.mark.skipif(not _gear_available(), reason="run python -m rl.tools.usd_gear")
def test_gear_is_visual_only_and_changes_no_physics():
    """The USD shells are inflated convex hulls, added for looks only.

    If this ever fails, the gear has acquired mass or collision and every
    result measured without it is no longer comparable.
    """
    def run(gear):
        sc = CS.build_scene(T.load_patch("B"), gear=gear)
        m, d = sc.model, sc.data
        ctrl = m.key(sc.key_id).ctrl.copy()
        for _ in range(500):
            d.ctrl[:] = ctrl
            sc.step()
        return sc.model, d.qpos.copy()

    m0, q0 = run(False)
    m1, q1 = run(True)
    assert m1.ngeom > m0.ngeom                       # gear really was added
    assert (m1.nq, m1.nv, m1.nu) == (m0.nq, m0.nv, m0.nu)
    assert m1.body_mass.sum() == pytest.approx(m0.body_mass.sum(), abs=1e-12)
    assert np.array_equal(q0, q1)                    # bit-identical trajectory


@pytest.mark.skipif(not _gear_available(), reason="run python -m rl.tools.usd_gear")
def test_gear_attaches_to_real_bodies():
    from rl.tools import usd_gear

    _, items = usd_gear.load_manifest()
    m = CS.build_scene(T.load_patch("B"), gear=True).model
    bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
    assert {it["body"] for it in items} <= bodies
    # the ascender tool rides the right wrist, where the grip constraint acts
    assert any(it["name"] == "ascender" and it["body"] == "right_wrist_yaw_link"
               for it in items)


# -- robot swap: assets/robots/mujoco vs the playground reference -----------

from rl.environment import robot as R  # noqa: E402


@pytest.mark.parametrize("name", ["playground", "himalaya"])
def test_scene_builds_on_either_robot(name):
    sc = CS.build_scene(T.load_patch("B"), robot_scene=R.resolve(name).xml)
    m = sc.model
    assert (m.nq, m.nv, m.nu) == (36, 35, 29)
    assert m.nmocap == 1 and m.neq == 1 and m.nhfield == 1
    assert sc.hand_rope_distance() < 1e-6


def test_himalaya_carries_the_ascender_mass():
    """+0.1 kg on the right wrist is the documented difference; nothing else."""
    pg = CS.build_scene(T.load_patch("B"), robot_scene=R.resolve("playground").xml).model
    hm = CS.build_scene(T.load_patch("B"), robot_scene=R.resolve("himalaya").xml).model
    assert hm.body_mass.sum() == pytest.approx(pg.body_mass.sum() + 0.1, abs=1e-6)
    name = lambda m, i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    for i in range(pg.nbody):
        b = name(pg, i)
        j = mujoco.mj_name2id(hm, mujoco.mjtObj.mjOBJ_BODY, b)
        if b == "right_wrist_yaw_link":
            continue
        assert hm.body_mass[j] == pytest.approx(pg.body_mass[i], abs=1e-9), b


def test_actuator_order_matches_the_policy_on_both_robots():
    """A permuted actuator order would silently scramble every joint command."""
    names = []
    for n in ("playground", "himalaya"):
        m = CS.build_scene(T.load_patch("B"), robot_scene=R.resolve(n).xml).model
        names.append([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(m.nu)])
    assert names[0] == names[1]


def test_observation_is_identical_across_robots():
    """Same state in, same 103 dims out -- otherwise the warm start is void.

    This caught a synthesised `local_linvel_pelvis` built as a framelinvel
    sensor referenced to its own site, which reads a constant zero. Playground
    declares it as a velocimeter.
    """
    from rl.environment import walk_policy as WP

    flat = T.make_terrain(0, 0.0, 1)
    qvel = np.random.default_rng(0).normal(0, 0.3, 35)
    obs = []
    for n in ("playground", "himalaya"):
        sc = CS.build_scene(flat, robot_scene=R.resolve(n).xml)
        m, d = sc.model, sc.data
        mujoco.mj_resetDataKeyframe(m, d, m.key("knees_bent").id)
        d.qvel[:] = qvel
        mujoco.mj_forward(m, d)
        obs.append(WP.WalkController(m).observe(d))
    assert np.allclose(obs[0], obs[1], atol=1e-12)


def test_spawn_fitter_finds_feet_on_both_robots():
    """Their foot geoms are unnamed, so name-matching buried the robot 6 cm."""
    for n in ("playground", "himalaya"):
        sc = CS.build_scene(T.make_terrain(0, 0.0, 1), robot_scene=R.resolve(n).xml)
        m, d = sc.model, sc.data
        feet = set(R.foot_contact_geoms(m))
        assert len(feet) >= 2
        touching = {d.contact[i].geom1 for i in range(d.ncon)} | {
            d.contact[i].geom2 for i in range(d.ncon)
        }
        # only the feet and the terrain may touch at reset
        assert (touching - feet) <= {m.geom("floor").id}
        assert min((d.contact[i].dist for i in range(d.ncon)), default=0) > -1e-3


def test_policy_compat_is_what_makes_the_walking_policy_transfer():
    from rl.environment import walk_policy as WP

    def upright_after(compat):
        sc = CS.build_scene(T.make_terrain(0, 0.0, 1),
                            robot_scene=R.resolve("himalaya").xml,
                            policy_compat=compat)
        m, d = sc.model, sc.data
        m.eq_active0[m.equality("ascender_grip").id] = 0
        sc.reset()
        ctl = WP.WalkController(m)
        for _ in range(int(6.0 / m.opt.timestep)):
            ctl.substep(d)
            sc.step()
        return float(d.site_xmat[m.site("imu_in_pelvis").id].reshape(3, 3)[2, 2])

    assert upright_after(True) > 0.9     # stands, as on the playground robot
    assert upright_after(False) < 0.6    # stock plant: the policy does not hold


# -- slope-aware reset pose ------------------------------------------------

@pytest.mark.parametrize("patch", ["B_slope25", "B", "B_slope45", "B_slope50"])
def test_reset_pose_keeps_ankles_inside_their_travel(patch):
    """Upright on a steep slope is kinematically impossible, not merely hard.

    The ankle would have to absorb the whole slope angle; past ~29 deg it hits
    its -50 deg stop and the robot balances on a foot edge, which no policy can
    hold. The reset pose splits the angle between a base lean and the ankles.
    """
    sc = CS.build_scene(T.load_patch(patch), robot_scene=R.resolve("himalaya").xml)
    assert sc.ankle_rad >= CS.ANKLE_PITCH_MIN
    assert sc.lean_rad >= 0.0
    # base lean plus ankle travel must cover the slope
    covered = np.degrees(sc.lean_rad) + (
        np.degrees(CS.KNEES_BENT_ANKLE - sc.ankle_rad)
    )
    assert covered == pytest.approx(sc.terrain.slope_deg, abs=1.0)


def test_flat_ground_reset_pose_is_unchanged():
    """No lean, no ankle change, when there is no slope to absorb."""
    sc = CS.build_scene(T.make_terrain(0, 0.12, 1), robot_scene=R.resolve("himalaya").xml)
    assert sc.lean_rad == pytest.approx(0.0)
    assert sc.ankle_rad == pytest.approx(CS.KNEES_BENT_ANKLE)


def test_policy_default_pose_is_the_checkpoint_not_the_scene_keyframe():
    """The scene's reset pose leans on a slope; the policy's origin must not.

    Reading default_pose off the scene keyframe would move the policy's
    operating point with the terrain and silently change what its actions mean.
    """
    from rl.environment import walk_policy as WP

    flat = CS.build_scene(T.make_terrain(0, 0.12, 1), robot_scene=R.resolve("himalaya").xml)
    steep = CS.build_scene(T.load_patch("B_slope45"), robot_scene=R.resolve("himalaya").xml)
    a = WP.WalkController(flat.model).default_pose
    b = WP.WalkController(steep.model).default_pose
    assert np.array_equal(a, b)
    assert np.allclose(a, R.KNEES_BENT_QPOS[7:36])
    # and the scene keyframes really do differ
    assert not np.allclose(flat.model.key("knees_bent").qpos,
                           steep.model.key("knees_bent").qpos)


def test_rope_meets_the_hand_at_every_slope():
    for patch in ("B_slope25", "B", "B_slope45", "B_slope50"):
        sc = CS.build_scene(T.load_patch(patch), robot_scene=R.resolve("himalaya").xml)
        assert sc.hand_rope_distance() < 1e-6, patch


def test_zero_friction_is_floored_rather_than_diverging():
    """condim=3 with friction 0 is a degenerate cone; it NaNs in ~30 ms.

    The failure surfaces at the lightest DOF (a wrist) and looks like a
    rope-attachment fault, so it is worth pinning down: it reproduces with the
    grip disabled, on flat ground, with no policy running.
    """
    ice = CS.FrictionParams.from_scalar(0.0)
    assert ice.clamped and ice.foot >= CS.MIN_FRICTION and ice.terrain >= CS.MIN_FRICTION
    assert not CS.FrictionParams.from_scalar(0.5).clamped

    sc = CS.build_scene(T.load_patch("B_slope25"),
                        robot_scene=R.resolve("himalaya").xml,
                        friction=CS.FrictionParams.from_scalar(0.0))
    m, d = sc.model, sc.data
    ctrl = m.key(sc.key_id).ctrl.copy()
    for _ in range(1500):
        d.ctrl[:] = ctrl
        sc.step()
    assert np.isfinite(d.qpos).all()
    assert np.abs(d.qacc).max() < 1e7


# -- the flat reference patch ----------------------------------------------

def test_flat_curriculum_patch_exists_and_is_flat():
    tr = T.load_patch("B_slope0")
    assert tr.slope_deg < 1.0
    assert 0.08 < tr.rough.std() < 0.15          # same roughness family
    assert tr.meta["slope_is_real"] is False     # honestly labelled
    assert "WARNING" in tr.meta


def test_unknown_patch_error_lists_what_exists():
    with pytest.raises(FileNotFoundError) as e:
        T.load_patch("B_slope99")
    msg = str(e.value)
    assert "B_slope0" in msg and "B_slope45" in msg and "--slope 0" in msg


def test_policy_stands_on_the_flat_patch_but_not_on_ice():
    """Friction, not the policy, is what a near-frictionless flat run tests.

    At mu<=0.05 the robot cannot stand on level ground either -- there is
    nothing to push against. This is the run that makes it look like the policy
    is broken when it is not.
    """
    from rl.environment import walk_policy as WP

    def upright_after(mu):
        sc = CS.build_scene(T.load_patch("B_slope0"),
                            robot_scene=R.resolve("himalaya").xml,
                            friction=CS.FrictionParams.from_scalar(mu))
        m, d = sc.model, sc.data
        m.eq_active0[m.equality("ascender_grip").id] = 0
        sc.reset()
        ctl = WP.WalkController(m)
        for _ in range(int(15.0 / m.opt.timestep)):
            ctl.substep(d)
            sc.step()
        assert np.isfinite(d.qpos).all()
        return float(d.site_xmat[m.site("imu_in_pelvis").id].reshape(3, 3)[2, 2])

    assert upright_after(0.9) > 0.9      # stands
    assert upright_after(0.0) < 0.6      # cannot stand on a frictionless floor


def test_policy_tracks_forward_command_in_the_body_frame():
    """The command is body-frame; yaw drifts, so world-frame x means nothing."""
    from rl.environment import walk_policy as WP

    def body_vx(cmd):
        sc = CS.build_scene(T.make_terrain(0, 0.12, 1),
                            robot_scene=R.resolve("himalaya").xml)
        m, d = sc.model, sc.data
        m.eq_active0[m.equality("ascender_grip").id] = 0
        sc.reset()
        ctl = WP.WalkController(m, command=(cmd, 0.0, 0.0))
        adr = m.sensor_adr[m.sensor("local_linvel_pelvis").id]
        vs = []
        for k in range(int(14.0 / m.opt.timestep)):
            ctl.substep(d)
            sc.step()
            if k >= int(6.0 / m.opt.timestep):
                vs.append(float(d.sensordata[adr]))
        return float(np.mean(vs))

    assert abs(body_vx(0.0)) < 0.15
    assert body_vx(1.0) > 0.4


def test_standing_friction_threshold_is_where_it_is_documented():
    """Below STAND_FRICTION the robot cannot stand on level ground, period.

    Pins the constant the CLI warns against, so the warning cannot drift away
    from the behaviour it describes.
    """
    from rl.environment import walk_policy as WP

    def upright_after(mu):
        sc = CS.build_scene(T.load_patch("B_slope0"),
                            robot_scene=R.resolve("himalaya").xml,
                            friction=CS.FrictionParams.from_scalar(mu))
        m, d = sc.model, sc.data
        sc.reset()
        ctl = WP.WalkController(m)
        for _ in range(int(15.0 / m.opt.timestep)):
            ctl.substep(d)
            sc.step()
        return float(d.site_xmat[m.site("imu_in_pelvis").id].reshape(3, 3)[2, 2])

    assert upright_after(CS.STAND_FRICTION + 0.1) > 0.9    # comfortably above
    assert upright_after(CS.MIN_FRICTION) < 0.6            # an ice rink
