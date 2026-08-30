"""Put the two gitignored asset trees the merged scene needs on disk.

`rl/environment/robot.py` and `assets/robots/mujoco/g1_unitree_ascender.xml`
both expect trees that a fetch step is supposed to create, and on this machine
neither fetch works:

  `<repo>/.reference/`  -- mujoco_playground's G1 scene, read by
      `robot.PLAYGROUND_SCENE`. `rl/tools/fetch_reference_model.py` downloads it
      from GitHub and dies here with
      `URLError: [SSL: CERTIFICATE_VERIFY_FAILED]`. It also writes to
      `rl/.reference` (`__file__.parent.parent`) while `robot.py` reads
      `<repo>/.reference` -- so even a successful fetch lands in the wrong
      place. Both reported as gaps.

  `assets/robots/g1/_menagerie/unitree_g1/assets/`  -- the stock Unitree link
      STLs the jacketed robot's meshes point at.
      `assets/robots/mujoco/build.py --fetch` is the documented step and dies
      with `ModuleNotFoundError: No module named 'pxr'`, because it imports
      `../g1/build_g1_usd.py`.

Both trees are byte-identical to files ALREADY INSTALLED in the venv:
mujoco_playground ships the G1 xmls, and its `external_deps/mujoco_menagerie`
is the same menagerie clone at the same pinned commit. So we copy from there:
no network, no `pxr`, nothing in anyone's source tree touched. Both targets are
gitignored (`.reference/`, `_menagerie/`).

Idempotent and cheap -- safe to call at the top of every run.
"""

import os
import shutil

_HARNESS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(_HARNESS_DIRECTORY))
REFERENCE_DIRECTORY = os.path.join(REPOSITORY_ROOT, ".reference")
HIMALAYA_MENAGERIE_DIRECTORY = os.path.join(
    REPOSITORY_ROOT, "assets", "robots", "g1", "_menagerie", "unitree_g1", "assets")


def _playground_xml_directory():
    from mujoco_playground._src.locomotion.g1 import g1_constants as constants
    return os.path.dirname(constants.task_to_xml("flat_terrain").as_posix())


def _menagerie_asset_directory():
    from mujoco_playground._src import mjx_env
    return (mjx_env.MENAGERIE_PATH / "unitree_g1" / "assets").as_posix()


def ensure_playground_reference(verbose=True) -> str:
    """`<repo>/.reference/` <- the installed playground G1 scene. -> the dir."""
    marker = os.path.join(REFERENCE_DIRECTORY, "scene_mjx_feetonly_flat_terrain.xml")
    if os.path.isfile(marker):
        return REFERENCE_DIRECTORY
    source = _playground_xml_directory()
    os.makedirs(REFERENCE_DIRECTORY, exist_ok=True)
    copied = 0
    for root, _, filenames in os.walk(source):
        relative = os.path.relpath(root, source)
        destination = (REFERENCE_DIRECTORY if relative == "."
                       else os.path.join(REFERENCE_DIRECTORY, relative))
        os.makedirs(destination, exist_ok=True)
        for filename in filenames:
            shutil.copy2(os.path.join(root, filename),
                         os.path.join(destination, filename))
            copied += 1
    # The copied XMLs reach menagerie meshes via a 5-up relative path
    # (`../../../../../mujoco_menagerie/...`) that only resolves when the
    # repo sits deep enough. Rewrite them to the ABSOLUTE vendored
    # menagerie path -- the same commit mujoco_playground vendors, so the
    # STLs are byte-identical wherever the repo is checked out.
    menagerie_root = os.path.dirname(
        os.path.dirname(_menagerie_asset_directory()))
    rewritten = 0
    for dirpath, _, filenames in os.walk(REFERENCE_DIRECTORY):
        for filename in filenames:
            if not filename.endswith(".xml"):
                continue
            xml_path = os.path.join(dirpath, filename)
            with open(xml_path, encoding="utf-8") as f:
                text = f.read()
            if "mujoco_menagerie/" not in text:
                continue
            import re
            text = re.sub(
                r"(?:\.\./)+mujoco_menagerie/", menagerie_root + "/", text
            )
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(text)
            rewritten += 1
    if verbose:
        print(f"[assets] .reference/ provisioned from the installed"
              f" mujoco_playground ({copied} files), mesh paths ->"
              f" {menagerie_root} -- the repo's own fetch"
              " tool cannot reach GitHub from here", flush=True)
    return REFERENCE_DIRECTORY


def ensure_himalaya_menagerie(verbose=True) -> str:
    """The jacketed robot's stock link STLs. -> the assets dir."""
    marker = os.path.join(HIMALAYA_MENAGERIE_DIRECTORY, "pelvis.STL")
    if os.path.isfile(marker):
        return HIMALAYA_MENAGERIE_DIRECTORY
    source = _menagerie_asset_directory()
    os.makedirs(HIMALAYA_MENAGERIE_DIRECTORY, exist_ok=True)
    copied = 0
    for filename in os.listdir(source):
        if filename.lower().endswith(".stl"):
            shutil.copy2(os.path.join(source, filename),
                         os.path.join(HIMALAYA_MENAGERIE_DIRECTORY, filename))
            copied += 1
    if verbose:
        print(f"[assets] assets/robots/g1/_menagerie provisioned from the"
              f" installed mujoco_menagerie ({copied} STLs) -- build.py --fetch"
              " needs usd-core and fails here", flush=True)
    return HIMALAYA_MENAGERIE_DIRECTORY


HOME_MENAGERIE_DIRECTORY = os.path.join(
    os.path.expanduser("~"), "mujoco_menagerie")


def ensure_home_menagerie(verbose=True) -> str:
    """`~/mujoco_menagerie` <- the installed clone, as a symlink.

    `rl/scripts/climb_scene.py --check` and the `--visual` paths resolve stock
    link meshes through the user's home clone. Symlinked, not copied: it is the
    same commit mujoco_playground already vendored, and a 34 MB duplicate in
    $HOME is not something a harness should leave behind.
    """
    if os.path.exists(HOME_MENAGERIE_DIRECTORY):
        return HOME_MENAGERIE_DIRECTORY
    from mujoco_playground._src import mjx_env
    source = mjx_env.MENAGERIE_PATH.as_posix()
    os.symlink(source, HOME_MENAGERIE_DIRECTORY)
    if verbose:
        print(f"[assets] ~/mujoco_menagerie -> {source} (symlink)", flush=True)
    return HOME_MENAGERIE_DIRECTORY


def ensure_all(verbose=True) -> None:
    ensure_playground_reference(verbose)
    ensure_himalaya_menagerie(verbose)
    ensure_home_menagerie(verbose)


if __name__ == "__main__":
    ensure_all()
