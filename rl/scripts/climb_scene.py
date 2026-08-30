#!/usr/bin/env python3
"""Build, inspect, export and view the merged G1 + terrain + rope scene.

    python -m rl.scripts.climb_scene --list
    python -m rl.scripts.climb_scene --patch B --export build/climb_B.xml
    python -m rl.scripts.climb_scene --slope 45 --ice 0.2 --wind 20 --check
    python -m rl.scripts.climb_scene --patch B --view

`--check` runs the headless acceptance suite: terrain surface agreement,
grip integrity, the fall-arrest comparison and a sweep of all four
randomisation axes. It is the fastest way to confirm the merge is sound after
a change.
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

from rl.environment import climb_scene as CS
from rl.environment import terrain as T
from rl.environment import robot as R
from rl.environment import walk_policy as WP

ROPE_RADIUS = CS.RopeParams().radius




def make_terrain(a) -> "T.Terrain":
    if a.slope is not None:
        return T.make_terrain(a.slope, a.rough, a.seed)
    return (T.load_patch_baked if a.baked else T.load_patch)(a.patch)


def robot_scene(a) -> str:
    """This project's G1 by default; --robot playground for the reference model.

    `--visual` only means anything for the playground robot, whose default
    fixture is mesh-stripped. The himalaya MJCF ships its own meshes.
    """
    return R.resolve(a.robot, visual=a.visual).xml


def build(a) -> CS.ClimbScene:
    return CS.build_scene(
        make_terrain(a),
        robot_scene=robot_scene(a),
        gear=a.gear,
        policy_compat=not a.stock_plant,
        lean_frac=a.lean_frac,
        friction=CS.FrictionParams.from_scalar(a.friction),
        rope=CS.RopeParams(n_waypoints=a.waypoints, collide=not a.rope_scenery),
        ratchet=not a.no_ratchet,
        spawn_frac=a.spawn_frac,
    )


def haul_force(sc: CS.ClimbScene, newtons: float):
    """A force up the fall line, to demonstrate the ascender sliding.

    The walking policy cannot climb -- on a slope it sags onto the rope and the
    hand legitimately stays put, which looks identical to a seized ascender.
    Hauling the base up the line drives the hand along the rope so the mechanism
    is visible.
    """
    if not newtons:
        return None
    pelvis = sc.model.body("pelvis").id

    def apply(data):
        tangent = sc.route.tangent_at(sc.ascender.s)
        data.xfrc_applied[pelvis, :3] = newtons * tangent

    return apply


def make_controller(sc: CS.ClimbScene, a):
    """The mels walking policy, or None to hold the keyframe pose open-loop.

    Holding the keyframe is not a controller: 29 position servos on fixed
    targets with no balance feedback. It topples on flat ground in about a
    second, so a fall in that mode says nothing about the terrain or the rope.
    """
    if a.no_policy:
        return None
    return WP.WalkController(sc.model, command=(a.cmd_x, a.cmd_y, a.cmd_yaw))


def apply_control(sc: CS.ClimbScene, ctl, keyframe_ctrl, haul=None) -> None:
    if ctl is None:
        sc.data.ctrl[:] = keyframe_ctrl
    else:
        ctl.substep(sc.data)
    if haul is not None:
        haul(sc.data)


def describe(sc: CS.ClimbScene) -> None:
    m, tr = sc.model, sc.terrain
    if sc.adapt_report["added"] or sc.adapt_report["retuned"]:
        for k in ("added", "retuned"):
            if sc.adapt_report[k]:
                print(f"robot     {k}: {', '.join(sc.adapt_report[k])}")
    print(f"terrain   {tr.source}  {tr.shape[1]}x{tr.shape[0]} @ {tr.res*100:.0f} cm")
    print(
        f"          slope {tr.slope_deg:.2f} deg   roughness rms {tr.rough.std():.3f} m"
        f"   world x extent +-{tr.world_extent_x:.2f} m"
    )
    if tr.meta.get("slope_is_real") is False:
        print(
            f"          !! curriculum patch: slope OVERRIDDEN to "
            f"{tr.meta.get('applied_slope_deg')} deg (real here "
            f"{tr.meta.get('real_slope_deg')} deg) -- a training aid, not a measurement"
        )
    print(f"rope      {sc.route.n_seg} segments, {sc.route.length:.2f} m")
    print(f"friction  terrain {sc.friction.terrain}  foot {sc.friction.foot}"
          f" (geoms, contact pairs and priority geoms)")
    if sc.friction.clamped:
        print(f"          raised to the {CS.MIN_FRICTION} floor -- friction exactly 0 is a "
              f"degenerate\n          contact cone and the solver diverges in ~30 ms")
    if sc.friction.foot < CS.STAND_FRICTION:
        print(f"  !!      {sc.friction.foot} is BELOW {CS.STAND_FRICTION}, the least this robot "
              f"needs to stand on LEVEL\n          ground. It will slip and fall whatever "
              f"the slope, policy or rope --\n          physics, not a policy fault. Use "
              f"--friction 0.3 or more to exercise the policy.")
    print(
        f"model     nq={m.nq} nv={m.nv} nu={m.nu} nmocap={m.nmocap} neq={m.neq}"
        f" ngeom={m.ngeom}"
    )
    print(f"spawn     {np.round(sc.spawn, 3)}   hand-rope gap {sc.hand_rope_distance():.2e} m")
    print(f"reset     base lean {np.degrees(sc.lean_rad):.1f} deg, "
          f"ankle pitch {np.degrees(sc.ankle_rad):.1f} deg "
          f"(upright on this slope would need "
          f"{np.degrees(CS.KNEES_BENT_ANKLE) - tr.slope_deg:.1f} deg, "
          f"limit {np.degrees(CS.ANKLE_PITCH_MIN):.0f})")


def simulate(sc: CS.ClimbScene, secs: float, wind: CS.WindParams, grip=True, ctl=None, haul=None):
    m, d = sc.model, sc.data
    if not grip:
        m.eq_active0[m.equality("ascender_grip").id] = 0
    sc.reset()
    if ctl is not None:
        ctl.reset()
    ctrl = m.key(sc.key_id).ctrl.copy()
    z0, s0 = float(d.qpos[2]), sc.ascender.s
    for _ in range(int(secs / m.opt.timestep)):
        apply_control(sc, ctl, ctrl, haul)
        sc.step(wind)
    upright = float(d.site_xmat[m.site("imu_in_pelvis").id].reshape(3, 3)[2, 2])
    return dict(
        upright=upright,
        dz=float(d.qpos[2]) - z0,
        ds=sc.ascender.s - s0,
        gap=sc.hand_rope_distance(),
        ncon=int(d.ncon),
        finite=bool(np.isfinite(d.qpos).all()),
    )


def check(a) -> int:
    ok = True
    print("=" * 72)
    print("1. terrain surface: MuJoCo heightfield vs terrain.surface_z()")
    mask = np.zeros(6, np.uint8)
    mask[CS.GROUP_TERRAIN] = 1
    for tr, label in [
        (T.make_terrain(25, 0.12, 1), "synth 25 deg"),
        (T.make_terrain(50, 0.12, 2), "synth 50 deg"),
        (T.load_patch("B"), "patch B separable"),
        (T.load_patch_baked("B"), "patch B baked"),
    ]:
        sc = CS.build_scene(tr)
        ext = tr.world_extent_x - 0.5
        errs = []
        for x in np.linspace(-ext, ext, 21):
            for y in (-3.0, 0.0, 3.0):
                top = float(tr.surface_z(x, y)) + 30.0
                g = np.zeros(1, np.int32)
                dist = mujoco.mj_ray(
                    sc.model, sc.data, np.array([x, y, top]),
                    np.array([0.0, 0.0, -1.0]), mask, 1, -1, g,
                )
                if mujoco.mj_id2name(sc.model, mujoco.mjtObj.mjOBJ_GEOM, int(g[0])) != "floor":
                    continue
                errs.append(top - dist - float(tr.surface_z(x, y)))
        e = np.abs(np.array(errs))
        good = e.max() < 0.05
        ok &= good
        print(f"   {label:<22} max err {e.max()*1000:7.2f} mm   {'OK' if good else 'FAIL'}")

    print(f"2. grip integrity at reset (hand must start exactly on the line)")
    sc = build(a)
    good = sc.hand_rope_distance() < 1e-6
    ok &= good
    print(f"   gap {sc.hand_rope_distance():.2e} m   {'OK' if good else 'FAIL'}")

    print("3. fall arrest: roped vs unroped, zero policy")
    roped = simulate(build(a), 4.0, CS.WindParams())
    unroped = simulate(build(a), 4.0, CS.WindParams(), grip=False)
    # The property is "stays on the line and descends materially less", not a
    # particular ratio: the himalaya robot has full-body collision and catches
    # on the terrain, so it falls less far unroped than the feet-only
    # playground model does. Asserting a fixed multiple tested the wrong thing.
    # "Still on the rope" means inside its radius, not inside a millimetre: the
    # carrier is a real sliding body now, so the grip has a few mm of compliance
    # where the welded mocap version had none.
    good = (
        roped["finite"]
        and roped["gap"] < ROPE_RADIUS    # still on the line
        and unroped["gap"] > 0.25         # came off it
        and abs(roped["dz"]) < 0.6 * abs(unroped["dz"])
    )
    ok &= good
    print(f"   roped   dz={roped['dz']:+7.2f} m  gap={roped['gap']:.2e} m  finite={roped['finite']}")
    print(f"   unroped dz={unroped['dz']:+7.2f} m  gap={unroped['gap']:.2f} m")
    print(f"   arrested: {'OK' if good else 'FAIL'}")

    print("4. ratchet: the hand must never slide back down the line")
    sc = build(a)
    m, d = sc.model, sc.data
    ctrl = m.key(sc.key_id).ctrl.copy()
    s_prev, viol = sc.ascender.s, 0.0
    for _ in range(2000):
        d.ctrl[:] = ctrl
        s = sc.step()
        viol = max(viol, s_prev - s)
        s_prev = s
    good = viol < 1e-12
    ok &= good
    print(f"   worst backslide {viol:.2e} m   {'OK' if good else 'FAIL'}")

    print("5. randomisation axes must each change the outcome")
    def spread(vals):
        return max(vals) - min(vals)

    # Friction is asserted directly on the contact value the solver uses, not
    # on a trajectory statistic: an uncontrolled robot tumbling down a 39 deg
    # face is chaotic, and its slide distance is not even monotonic in mu. The
    # direct check is also the one that catches the two ways this axis silently
    # dies -- an explicit <pair> overriding geom friction, and priority-1 foot
    # geoms overriding the terrain.
    friction_ok = True
    for mu in (0.9, 0.4, 0.05):
        sc_i = CS.build_scene(
            make_terrain(a), robot_scene=robot_scene(a),
            friction=CS.FrictionParams.from_scalar(mu), policy_compat=not a.stock_plant,
        )
        m_i, d_i = sc_i.model, sc_i.data
        m_i.eq_active0[m_i.equality("ascender_grip").id] = 0
        sc_i.reset()
        feet = set(R.foot_contact_geoms(m_i))
        kc = m_i.key(sc_i.key_id).ctrl.copy()
        seen = set()
        for _ in range(1000):
            d_i.ctrl[:] = kc
            sc_i.step()
            for i in range(d_i.ncon):
                c = d_i.contact[i]
                if c.geom1 in feet or c.geom2 in feet:
                    seen.add(round(float(c.friction[0]), 4))
        hit = any(abs(v - mu) < 1e-6 for v in seen)
        friction_ok &= hit
        print(f"   friction mu={mu:<5} foot-contact mu seen {sorted(seen)}  "
              f"{'OK' if hit else 'FAIL'}")
    ok &= friction_ok
    # Wind is asserted on the force actually written to the torso. Its effect on
    # the roped sag is small now that a solid rope holds the robot against the
    # slope, so that trajectory statistic no longer separates the settings --
    # same reason friction is checked directly.
    wind_ok = True
    forces = []
    for w in (0.0, 10.0, 30.0):
        sc_w = build(a)
        sc_w.reset()
        sc_w.apply_wind(CS.WindParams(speed=w, heading=np.pi))
        forces.append(float(np.linalg.norm(sc_w.data.xfrc_applied[sc_w.torso_body_id, :3])))
    wind_ok = forces[0] == 0.0 and forces[1] > 1.0 and forces[2] > 4 * forces[1]
    ok &= wind_ok
    print(f"   wind     torso force at 0/10/30 m/s: "
          f"{forces[0]:.1f} / {forces[1]:.1f} / {forces[2]:.1f} N   "
          f"{'OK' if wind_ok else 'FAIL'}")
    slope_r = [
        simulate(CS.build_scene(T.make_terrain(s, 0.12, 1)), 3.0, CS.WindParams())["dz"]
        for s in (25, 50)
    ]
    seed_r = [
        simulate(CS.build_scene(T.make_terrain(38, 0.12, sd)), 3.0, CS.WindParams())["dz"]
        for sd in (1, 7)
    ]
    for name, vals, tol in [
        ("slope    (roped sag)", slope_r, 0.01),
        ("surface  (roped sag)", seed_r, 0.001),
    ]:
        good = spread(vals) > tol
        ok &= good
        print(f"   {name:<26} spread {spread(vals):7.3f} m   {'OK' if good else 'FAIL'}")

    print("=" * 72)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def render(sc: CS.ClimbScene, prefix: str, wind: CS.WindParams, ctl=None) -> None:
    """Write preview stills. Cameras are placed and checked for clearance.

    The slope rises toward +x, so a camera at an uphill azimuth ends up buried
    inside the mountain; each shot asserts it is above the surface first.
    """
    from PIL import Image

    m, d, tr = sc.model, sc.data, sc.terrain
    gid = m.geom("floor").id
    m.geom_matid[gid] = -1
    m.geom_rgba[gid] = [0.90, 0.93, 0.97, 1.0]
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    ctrl = m.key(sc.key_id).ctrl.copy()
    r = mujoco.Renderer(m, 820, 1280)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    shots = [
        ("start", 0.0, 55, -12, 3.2, None),
        ("hang", 3.0, 55, -12, 3.2, None),
        ("wide", 0.0, 55, -14, 12.0, None),
        ("route", 0.0, 40, -18, 26.0, [0.0, 0.0, 0.4]),
    ]
    for tag, secs, az, el, dist, lookat in shots:
        sc.reset()
        if ctl is not None:
            ctl.reset()
        for _ in range(int(secs / m.opt.timestep)):
            apply_control(sc, ctl, ctrl)
            sc.step(wind)
        la = np.array(lookat) if lookat is not None else d.qpos[:3].copy()
        ar, er = np.radians(az), np.radians(el)
        cp = la + dist * np.array(
            [-np.cos(er) * np.cos(ar), -np.cos(er) * np.sin(ar), -np.sin(er)]
        )
        clear = cp[2] - float(tr.surface_z(cp[0], cp[1]))
        cam.lookat[:] = la
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        r.update_scene(d, cam, opt)
        path = f"{prefix}_{tag}.png"
        Image.fromarray(r.render()).save(path)
        flag = "" if clear > 0.5 else "   !! camera below the surface"
        print(f"   {path}  (camera clearance {clear:+.1f} m){flag}")


def view(sc: CS.ClimbScene, wind: CS.WindParams, ctl=None, passive: bool = False, haul=None) -> int:
    """Open the interactive viewer with the ascender ratchet running.

    macOS makes this awkward. `launch_passive` hands control back to the caller
    -- which is what you want, since the ratchet has to run every substep -- but
    it requires the script to be started under `mjpython`. `launch` runs fine
    under plain `python3`, but it owns the stepping loop and takes no per-step
    hook, so on its own the carrier would never move and the hand would be
    welded to a fixed point.

    `set_mjcb_control` is the way through: MuJoCo calls it from inside mj_step,
    after the position and velocity stages, so `site_xpos` is current when the
    palm is read. Writing `mocap_pos` there lands one substep later (2 ms, under
    a millimetre of palm travel), which the grip constraint does not notice.
    """
    from mujoco import viewer as mj_viewer

    m, d = sc.model, sc.data
    ctrl = m.key(sc.key_id).ctrl.copy()
    sc.reset()
    if ctl is not None:
        ctl.reset()

    if passive:
        with mj_viewer.launch_passive(m, d) as v:
            while v.is_running():
                apply_control(sc, ctl, ctrl, haul)
                sc.step(wind)
                v.sync()
        return 0

    def control_cb(model, data):
        apply_control(sc, ctl, ctrl, haul)
        if wind.speed:
            rel = wind.velocity - data.cvel[sc.torso_body_id, 3:5]
            data.xfrc_applied[sc.torso_body_id, :] = 0.0
            data.xfrc_applied[sc.torso_body_id, :2] = (
                wind.drag_coeff * np.linalg.norm(rel) * rel
            )

    # The carrier is held on the rope by a projection applied AFTER each
    # mj_step. mjcb_control runs *inside* the step, too early for that, so the
    # projection goes in a passive callback which MuJoCo invokes at the end of
    # the pipeline.
    def passive_cb(model, data):
        sc.ascender.constrain(data)

    mujoco.set_mjcb_control(control_cb)
    mujoco.set_mjcb_passive(passive_cb)
    try:
        mj_viewer.launch(m, d)
    finally:
        mujoco.set_mjcb_control(None)
        mujoco.set_mjcb_passive(None)
    return 0


def verify_policy(a) -> int:
    """Prove the walking policy is loaded and driving the robot.

    Everything here is a pass/fail with a number next to it, because "the robot
    fell" has at least four unrelated causes in this scene -- no policy running,
    a broken observation, an incompatible plant, or terrain the policy simply
    cannot handle -- and they are indistinguishable by eye.
    """
    import hashlib

    ok = True
    print("=" * 72)
    print(f"1. checkpoint  {WP.DEFAULT_POLICY}")
    if not os.path.exists(WP.DEFAULT_POLICY):
        print("   MISSING")
        return 1
    digest = hashlib.sha256(open(WP.DEFAULT_POLICY, "rb").read()).hexdigest()[:16]
    pol = WP.WalkPolicy()
    shapes = " ".join(f"{w.shape[0]}->{w.shape[1]}" for w in pol.w)
    print(f"   sha256:{digest}  {shapes}")
    good = pol.obs_dim == WP.OBS_DIM and pol.w[3].shape[1] == 2 * WP.N_JOINTS
    ok &= good
    print(f"   obs_dim {pol.obs_dim}, head {pol.w[3].shape[1]} (= 2 x 29 mean+logstd)"
          f"   {'OK' if good else 'FAIL'}")

    print("2. forward pass")
    o = np.zeros(WP.OBS_DIM)
    a1, a2 = pol(o), pol(o)
    big = pol(np.full(WP.OBS_DIM, 100.0))
    good = (np.array_equal(a1, a2) and a1.shape == (WP.N_JOINTS,)
            and np.isfinite(a1).all() and np.isfinite(big).all())
    ok &= good
    print(f"   deterministic, shape {a1.shape}, finite on extreme input"
          f"   {'OK' if good else 'FAIL'}")

    print("3. observation layout (playground g1/joystick _get_obs order)")
    flat = CS.build_scene(T.make_terrain(0, 0.12, 1), robot_scene=robot_scene(a),
                          policy_compat=not a.stock_plant)
    ctl = WP.WalkController(flat.model, pol, command=(0.3, 0.0, 0.0))
    obs = ctl.observe(flat.data)
    checks = [
        ("length 103", obs.shape == (WP.OBS_DIM,)),
        ("command block at 9:12", np.allclose(obs[9:12], [0.3, 0.0, 0.0])),
        ("last_act zeroed at reset", np.allclose(obs[70:99], 0.0)),
        ("phase = [cos,cos,sin,sin]", np.allclose(obs[99:103], [1, -1, 0, 0], atol=1e-9)),
        ("gravity points down", obs[8] < -0.9),
        ("default_pose = training knees_bent",
         np.allclose(ctl.default_pose, R.KNEES_BENT_QPOS[7:36])),
    ]
    for label, good in checks:
        ok &= good
        print(f"   {label:<38} {'OK' if good else 'FAIL'}")

    print("4. flat ground, 30 s: the policy must hold the robot up")
    for label, use_policy, want_up in (("mels policy", True, True),
                                       ("no policy (keyframe hold)", False, False)):
        sc = CS.build_scene(T.make_terrain(0, 0.12, 1), robot_scene=robot_scene(a),
                            policy_compat=not a.stock_plant,
                            rope=CS.RopeParams(collide=False))
        m, d = sc.model, sc.data
        m.eq_active0[m.equality("ascender_grip").id] = 0
        sc.reset()
        c = WP.WalkController(m, pol) if use_policy else None
        kc = m.key(sc.key_id).ctrl.copy()
        z0 = float(d.qpos[2])
        for _ in range(int(30.0 / m.opt.timestep)):
            apply_control(sc, c, kc)
            sc.step(CS.WindParams())
        up = float(d.site_xmat[m.site("imu_in_pelvis").id].reshape(3, 3)[2, 2])
        stood = up > 0.9 and abs(float(d.qpos[2]) - z0) < 0.15
        good = stood == want_up
        ok &= good
        print(f"   {label:<28} upright={up:+.2f} dz={float(d.qpos[2])-z0:+.3f} m"
              f"   {'OK' if good else 'FAIL'}")

    print("5. it tracks a velocity command")
    print("   (body frame: the command is forward velocity in the robot's own")
    print("    frame, and yaw drifts freely, so world-frame x is meaningless)")
    for cmd, lo, hi in ((0.0, None, 0.15), (0.6, 0.30, None), (1.0, 0.40, None)):
        # No rope collision here: this measures the policy, and a solid rope in
        # the path of a wandering robot confounds it.
        sc = CS.build_scene(T.make_terrain(0, 0.12, 1), robot_scene=robot_scene(a),
                            policy_compat=not a.stock_plant,
                            rope=CS.RopeParams(collide=False))
        m, d = sc.model, sc.data
        m.eq_active0[m.equality("ascender_grip").id] = 0
        sc.reset()
        c = WP.WalkController(m, pol, command=(cmd, 0.0, 0.0))
        adr = m.sensor_adr[m.sensor("local_linvel_pelvis").id]
        secs, tail = 14.0, 8.0
        n, n0 = int(secs / m.opt.timestep), int((secs - tail) / m.opt.timestep)
        vs = []
        for k in range(n):
            c.substep(d)
            sc.step(CS.WindParams())
            if k >= n0:
                vs.append(float(d.sensordata[adr]))
        vx = float(np.mean(vs))
        good = (lo is None or vx > lo) and (hi is None or abs(vx) < hi)
        ok &= good
        print(f"   cmd_x={cmd:.1f} -> body-frame vx {vx:+.2f} m/s   "
              f"{'OK' if good else 'FAIL'}")
    print("   note: commands below ~0.4 do not initiate a gait; the policy just")
    print("   stands. 0.66 m/s at cmd 1.0 matches the documented 0.75.")

    print("=" * 72)
    print("POLICY VERIFIED" if ok else "POLICY CHECKS FAILED")
    if ok:
        print("A fall on sloped terrain is therefore the policy's limits, not a"
              " loading fault:\n   it was trained on flat ground and has never"
              " seen an incline.")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch", default="B", help="shipped patch name (see --list)")
    p.add_argument("--baked", action="store_true", help="load the patch verbatim, slope baked into the grid")
    p.add_argument("--slope", type=float, default=None, help="synthesise terrain at this slope instead of loading a patch")
    p.add_argument("--rough", type=float, default=T.DEFAULT_ROUGH_RMS, help="roughness rms, m (surface-variation axis)")
    p.add_argument("--seed", type=int, default=0, help="roughness seed (surface-variation axis)")
    p.add_argument("--friction", "--ice", type=float, default=0.9, dest="friction",
                   help="foot+terrain sliding friction. 0.9 dry rock, 0.3 the least "
                        "the robot can stand on, 0.1 verglas, 0.02 bare ice. "
                        "(--ice is the old name and is NOT inverted: --ice 0 meant "
                        "zero friction, i.e. maximum slipperiness.)")
    p.add_argument("--wind", type=float, default=0.0, help="wind speed, m/s")
    p.add_argument("--wind-heading", type=float, default=np.pi, help="wind heading, rad (default pi = downslope)")
    p.add_argument("--waypoints", type=int, default=9)
    p.add_argument("--spawn-frac", type=float, default=0.12)
    p.add_argument("--no-ratchet", action="store_true", help="let the hand slide back down (ablation)")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--list", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--verify-policy", action="store_true",
                   help="prove the walking policy is loaded and driving the robot")
    p.add_argument("--export", metavar="XML")
    p.add_argument("--view", action="store_true")
    p.add_argument("--no-policy", action="store_true",
                   help="hold the keyframe pose open-loop instead of running the policy")
    p.add_argument("--cmd-x", type=float, default=0.0, help="forward velocity command, m/s")
    p.add_argument("--cmd-y", type=float, default=0.0, help="strafe command, m/s")
    p.add_argument("--cmd-yaw", type=float, default=0.0, help="yaw rate command, rad/s")
    p.add_argument("--passive", action="store_true",
                   help="use launch_passive instead; macOS requires `mjpython` for it")
    p.add_argument("--robot", default="himalaya",
                   choices=("himalaya", "himalaya-bare", "playground"),
                   help="himalaya = assets/robots/mujoco (jacket, boots, ascender); "
                        "playground = the model the mels policy was trained in")
    p.add_argument("--rope-scenery", action="store_true",
                   help="make the rope non-colliding (visual only), as it was before")
    p.add_argument("--haul", type=float, default=0.0, metavar="N",
                   help="force up the fall line, newtons. The walking policy cannot "
                        "climb, so use this to see the ascender slide. The robot "
                        "weighs 338 N, so ~211 N is needed just to hold station on "
                        "patch B; 230-300 climbs smoothly, 400+ tears the grip open.")
    p.add_argument("--lean-frac", type=float, default=None,
                   help="base lean as a fraction of the slope angle; default is the "
                        "smallest lean that keeps the ankles inside their travel")
    p.add_argument("--stock-plant", action="store_true",
                   help="keep this robot's own dynamics and 4-sphere feet instead of "
                        "matching playground's; the mels policy will not transfer")
    p.add_argument("--gear", action="store_true",
                   help="add the jacket/boots/ascender shells from g1_unitree.usd "
                        "(visual only; run rl.tools.usd_gear first)")
    p.add_argument("--visual", action="store_true",
                   help="use the mesh-bearing G1 (see rl.tools.fetch_visual_assets)")
    p.add_argument("--render", metavar="PREFIX",
                   help="write preview PNGs to PREFIX_{start,hang,wide,route}.png")
    a = p.parse_args(argv)

    if a.list:
        for cls, names in T.list_patches().items():
            print(f"{cls}:")
            for n in names:
                tr = T.load_patch(n)
                real = tr.meta.get("slope_is_real", True)
                tag = "REAL slope" if real else f"SYNTHETIC slope (real here {tr.meta.get('real_slope_deg')})"
                print(f"   {n:<14} {tr.slope_deg:5.2f} deg   {tag}")
        print("\nIn every patch all detail finer than ~30 m is synthetic.")
        return 0

    if a.verify_policy:
        return verify_policy(a)

    if a.check:
        return check(a)

    sc = build(a)
    describe(sc)

    if a.export:
        path = sc.export(a.export)
        print(f"\nexported {path}")
        print(f"         {path[:-4]}.hfield  (float32 heightfield)")
        m2 = mujoco.MjModel.from_xml_path(path)
        print(f"reloads OK: nq={m2.nq} nv={m2.nv} ngeom={m2.ngeom} nhfield={m2.nhfield}")

    wind = CS.WindParams(speed=a.wind, heading=a.wind_heading)

    if a.render:
        render(sc, a.render, wind, ctl=make_controller(sc, a))

    if a.view:
        return view(sc, wind, ctl=make_controller(sc, a),
                    passive=a.passive, haul=haul_force(sc, a.haul))

    ctl = make_controller(sc, a)
    label = "keyframe hold (NO policy)" if ctl is None else "mels walking policy"
    r = simulate(sc, a.seconds, wind, ctl=ctl, haul=haul_force(sc, a.haul))
    print(
        f"\n{a.seconds:.0f} s, {label}:  base dz={r['dz']:+.2f} m   upright={r['upright']:+.2f}   "
        f"ascender ds={r['ds']:+.2f} m   hand-rope gap={r['gap']:.2e} m   "
        f"contacts={r['ncon']}   finite={r['finite']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
