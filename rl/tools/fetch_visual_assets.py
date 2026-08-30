#!/usr/bin/env python3
"""Fetch the *renderable* G1: playground's XML plus the menagerie meshes.

`fetch_reference_model.py` deliberately strips meshes -- the deployment tests
only need kinematics, and it keeps that fixture a few kB. That model is perfect
for physics and useless to look at: the G1 renders as eight bare collision
capsules.

This fetches the unstripped model and sparse-checks-out just `unitree_g1` from
mujoco_menagerie (38 MB, versus several hundred for the whole repo), then
rewrites the mesh paths to point at the local copy. Physics is identical --
same nq/nv/nu, same sites, same keyframes -- so a scene built on it behaves
exactly like one built on `.reference/`, it just has a body.

    python -m rl.tools.fetch_visual_assets
    python -m rl.scripts.climb_scene --visual --render build/vis
"""
import pathlib
import subprocess
import sys
import urllib.request

PLAYGROUND_TAG = "v0.2.0"  # same pin as fetch_reference_model.py
BASE = (
    f"https://raw.githubusercontent.com/google-deepmind/mujoco_playground/"
    f"{PLAYGROUND_TAG}/mujoco_playground/_src/locomotion/g1/xmls"
)
FILES = ("g1_mjx_feetonly.xml", "scene_mjx_feetonly_flat_terrain.xml", "sensor.xml")
MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie.git"
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = ROOT / ".reference" / "full"
MEN_DIR = ROOT / ".reference" / "mujoco_menagerie"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as r:
            (OUT / name).write_text(r.read().decode())
        print(f"  {name}")

    if not (MEN_DIR / "unitree_g1" / "assets").is_dir():
        print(f"sparse-cloning unitree_g1 from menagerie into {MEN_DIR} ...")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1",
             MENAGERIE, str(MEN_DIR)],
            check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "unitree_g1"], cwd=MEN_DIR, check=True
        )
    else:
        print(f"menagerie already present at {MEN_DIR}")

    # The published XML points at ../../../../../mujoco_menagerie/... , which only
    # resolves inside a playground source tree.
    xml = OUT / "g1_mjx_feetonly.xml"
    text = xml.read_text()
    n = text.count("../../../../../mujoco_menagerie/")
    xml.write_text(text.replace("../../../../../mujoco_menagerie/", f"{MEN_DIR}/"))
    print(f"rewrote {n} mesh paths -> {MEN_DIR}")

    try:
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(OUT / FILES[1]))
        print(f"loads OK: nq={m.nq} nv={m.nv} nu={m.nu} nmesh={m.nmesh}")
    except ImportError:
        print("mujoco not installed here; assets written but not loaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
