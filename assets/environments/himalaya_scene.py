#!/usr/bin/env python3
"""Build the basic Himalaya scene with the Isaac Sim API: 3 m test pad + dressed G1 + lights + physics.

Run with Isaac Sim's python (host install or the container):
    ./python.sh assets/environments/himalaya_scene.py                 # GUI
    ./python.sh assets/environments/himalaya_scene.py --stream        # headless + WebRTC livestream (web viewer)
    ./python.sh assets/environments/himalaya_scene.py --headless --frames 300 --save out.usd   # smoke test / export

Container:  sudo docker exec -it isim-isaac-sim-1 ./python.sh /workspace/assets/environments/himalaya_scene.py --stream
"""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true")
ap.add_argument("--stream", action="store_true", help="headless + WebRTC livestream (same app as the web viewer)")
ap.add_argument("--public-ip", default=os.environ.get("ISAACSIM_HOST", "127.0.0.1"))
ap.add_argument("--signal-port", default=os.environ.get("ISAACSIM_SIGNAL_PORT", "49100"))
ap.add_argument("--stream-port", default=os.environ.get("ISAACSIM_STREAM_PORT", "47998"))
ap.add_argument("--frames", type=int, default=0, help="step N frames then exit (0 = run until closed)")
ap.add_argument("--save", default="", help="also save the assembled stage to this .usd")
ap.add_argument("--assets", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                help="assets/ root (defaults to this file's parent)")
args, _unknown = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402
if args.stream:
    # Boot the official streaming experience (isaacsim.exp.full.streaming) so the web viewer connects as usual.
    isaac_path = os.environ.get("ISAAC_PATH", "/isaac-sim")
    # hide_ui=False keeps the editor UI (menus, stage, property panels) in the stream; --no-window = no X window.
    simulation_app = SimulationApp(
        {"headless": False, "hide_ui": False, "extra_args": [
            f"--/exts/omni.kit.livestream.app/primaryStream/publicIp={args.public_ip}",
            f"--/exts/omni.kit.livestream.app/primaryStream/signalPort={args.signal_port}",
            f"--/exts/omni.kit.livestream.app/primaryStream/streamPort={args.stream_port}",
            "--no-window"]},
        experience=os.path.join(isaac_path, "apps", "isaacsim.exp.full.streaming.kit"))
else:
    simulation_app = SimulationApp({"headless": args.headless})

import omni.usd                                                     # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.utils.prims import create_prim                    # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage         # noqa: E402
from isaacsim.core.prims import Articulation                         # noqa: E402
from pxr import UsdGeom, UsdLux, Gf                                  # noqa: E402

ROBOT_USD = os.path.join(args.assets, "robots", "g1_unitree.usd")
TERRAIN_USD = os.path.join(args.assets, "environments", "himalaya_3m", "himalaya_3m.usd")
for f in (ROBOT_USD, TERRAIN_USD):
    assert os.path.exists(f), f"missing {f}"

world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=1 / 60)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

# terrain (collision + snow/ice/rock physics materials live in the USD)
add_reference_to_stage(TERRAIN_USD, "/World/ground")

# robot: articulation root sits with pelvis at z=0.793; face -X so it walks toward the slopes
add_reference_to_stage(ROBOT_USD, "/World/G1")
g1_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/G1"))
g1_xf.ClearXformOpOrder()
g1_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
g1_xf.AddRotateZOp().Set(180.0)

# lights: overcast high-altitude sky + low sun
dome = UsdLux.DomeLight.Define(stage, "/World/domeLight"); dome.CreateIntensityAttr(1000)
sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(3000); sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, 0))

robot = world.scene.add(Articulation(prim_paths_expr="/World/G1", name="g1"))
world.reset()
print(f"[himalaya_scene] G1: {robot.num_dof} DoF, pelvis z = {robot.get_world_poses()[0][0][2]:.3f} m")

if args.save:
    stage.GetRootLayer().Export(args.save); print("[himalaya_scene] saved", args.save)

i = 0
while simulation_app.is_running() and (args.frames == 0 or i < args.frames):
    world.step(render=True); i += 1
    if args.frames and i % 100 == 0:
        print(f"[himalaya_scene] frame {i}: pelvis z = {robot.get_world_poses()[0][0][2]:.3f} m")
simulation_app.close()
