"""Build the offline reference model the deployment tests validate against.

The deployment code must match the environment the policy was TRAINED in, so
the reference has to be pinned to a specific mujoco_playground version rather
than to whatever happens to be installed. The repo README pins playground
0.2.0, so PLAYGROUND_TAG below is v0.2.0.

(The G1 task is in fact unchanged between v0.0.2 and v0.2.0 -- _get_obs, the
actuator order, the knees_bent keyframe, action_scale, ctrl_dt and the gait
clock are all byte-identical, and the diffs are MJX-backend and collision
sensors. That was verified, not assumed, but it is luck rather than method:
pin the version so a future release cannot silently move the target.)

Menagerie mesh assets are stripped -- the tests only need kinematics, joint
axes and sites, none of which live in the meshes, and this keeps the fixture
a few kB instead of tens of MB.

    python tools/fetch_reference_model.py
    G1_NOMESH_SCENE=.reference/scene_mjx_feetonly_flat_terrain.xml \
        python -m pytest tests/test_deploy.py -q
"""
import pathlib
import re
import urllib.request

PLAYGROUND_TAG = "v0.2.0"
BASE = (f"https://raw.githubusercontent.com/google-deepmind/mujoco_playground/"
        f"{PLAYGROUND_TAG}/mujoco_playground/_src/locomotion/g1/xmls")
FILES = ("g1_mjx_feetonly.xml", "scene_mjx_feetonly_flat_terrain.xml",
         "sensor.xml")
OUT = pathlib.Path(__file__).resolve().parent.parent.parent / ".reference"


def strip_meshes(xml: str) -> str:
    """Drop mesh assets and mesh geoms; keep bodies, joints, sites, sensors."""
    xml = re.sub(r"<mesh\b[^>]*/>", "", xml)
    xml = re.sub(r'<geom\b[^>]*\bmesh="[^"]*"[^>]*/>', "", xml)
    return xml


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name in FILES:
        with urllib.request.urlopen(f"{BASE}/{name}") as r:
            text = r.read().decode()
        if name == "g1_mjx_feetonly.xml":
            text = strip_meshes(text)
        (OUT / name).write_text(text)
        print(f"  {name}  ({len(text)} bytes)")
    print(f"reference model ({PLAYGROUND_TAG}) written to {OUT}")

    try:
        import mujoco
    except ImportError:
        print("mujoco not installed here; fixture written but not loaded")
        return
    m = mujoco.MjModel.from_xml_path(
        str(OUT / "scene_mjx_feetonly_flat_terrain.xml"))
    print(f"loads OK: nq={m.nq} nv={m.nv} nu={m.nu}")


if __name__ == "__main__":
    main()
