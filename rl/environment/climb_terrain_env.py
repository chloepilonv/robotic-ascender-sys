"""G1 fixed-rope climb on the merged Lhotse terrain — MJX training env.

Reuses `climb_scene.build_scene` wholesale — the team's merged scene with
the real Lhotse Face heightfield (measured patches A-D or synthetic slope
overrides), the draped rope polyline, and a spawn pose fitted to the
surface — then adapts the result into the same shape the legacy
`G1ClimbAscender` env expects:

* The scene's 3-slide rope bead (`carrier_x/y/z`) is replaced by a single
  slide joint along the rope's mean uphill axis. That keeps
  `climb_env`'s jit/vmap-safe ratchet (a clamp on one qpos coordinate),
  the trimmed-observation machinery, and every reward helper working
  unchanged. The deviation between the draped polyline and its mean axis
  is the visual weave only — the physics line is the slide axis.
* Observations are bit-compatible with the legacy env and upstream G1
  Joystick (103-dim `state`, 216-dim `privileged_state`), so the mels
  policy loads unchanged.

Registered as `G1ClimbTerrain`; `terrain_config.patch` selects the
terrain (`B`, `A`-`D`, `B_flat0`, `B_slope30`, ...).
"""

import jax
import jax.numpy as jp
from mujoco_playground._src.locomotion.g1 import g1_constants as consts

from rl.environment import climb_env
from rl.environment import climb_scene
from rl.environment import terrain as terrain_mod
import functools
import math

from ml_collections import config_dict
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import locomotion
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick



def default_config() -> config_dict.ConfigDict:
  """Legacy climb config plus `terrain_config`."""
  cfg = climb_env.default_config()
  cfg.terrain_config = config_dict.create(
      # Terrain patch: measured Lhotse "A".."D", or synthetic overrides
      # like "B_slope30" / "B_flat0" (see terrain.list_patches()).
      patch="B",
      # Foot/terrain friction: retunes the explicit foot-floor contact
      # pairs (geom friction alone is overridden by those pairs).
      foot_friction=0.8,
  )
  return cfg


class G1ClimbTerrain(climb_env.G1ClimbAscender):
  """G1 fixed-rope climb on the Lhotse heightfield (MJX)."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict | None = None,
      config_overrides: dict | None = None,
  ):
    del task  # the terrain replaces the plane outright
    config = config or default_config()
    config.push_config.enable = False
    self._task = "flat_terrain"
    # Grandparent init (Joystick) for config/_post_init on the stock
    # scene, then swap the model for the merged terrain build.
    g1_joystick.Joystick.__init__(
        self, task=self._task, config=config, config_overrides=config_overrides
    )
    self._build_model()
    self._n_substeps = int(round(self.dt / self.sim_dt))

  def _build_model(self) -> None:
    cc = self._config.terrain_config
    terrain = terrain_mod.load_patch(cc.patch)

    # Merged scene: G1 + heightfield + draped rope + spawn pose fitted to
    # the surface. The scene's 3-slide bead becomes a single slide along
    # the rope's mean axis so the legacy env's single-coordinate ratchet
    # and obs trimming carry over unchanged.
    scene = climb_scene.build_scene(terrain=terrain)
    spec = scene.spec
    route = scene.route

    rope_vec = route.points[-1] - route.points[0]
    rope_len = float(np.linalg.norm(rope_vec))
    axis = rope_vec / rope_len

    for jn in ("carrier_x", "carrier_y", "carrier_z"):
      spec.delete(spec.joint(jn))
    carrier = spec.body("rope_carrier")
    carrier.add_joint(
        name="ascender_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=axis,
        damping=1.0,
        frictionloss=0.2,
        range=(-0.5, rope_len),
    )
    # Every keyframe: 36 robot qpos (7 freejoint + 29 joints) + slide 0.
    for k in spec.keys:
      k.qpos = np.concatenate([np.asarray(k.qpos)[:36], [0.0]])

    self._mj_model = spec.compile()
    self._mj_model.opt.timestep = self.sim_dt
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = consts.task_to_xml(self._task).as_posix()

    self._init_q = jp.array(self._mj_model.keyframe("knees_bent").qpos)
    self._slide_qposadr = int(
        self._mj_model.jnt_qposadr[self._mj_model.joint("ascender_slide").id]
    )
    self._slide_dofadr = int(
        self._mj_model.jnt_dofadr[self._mj_model.joint("ascender_slide").id]
    )
    assert self._slide_qposadr == self._mj_model.nq - 1
    assert self._slide_dofadr == self._mj_model.nv - 1
    self._default_pose = jp.array(
        self._mj_model.keyframe("knees_bent").qpos[7 : self._slide_qposadr]
    )
    self._lowers, self._uppers = (
        jp.array(self._mj_model.jnt_range[1 : self._mj_model.njnt - 1].T)
    )
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    f = self._config.soft_joint_pos_limit_factor
    self._soft_lowers = c - 0.5 * r * f
    self._soft_uppers = c + 0.5 * r * f

    self._palm_site_id = self._mj_model.site("right_palm").id
    self._carrier_site_id = self._mj_model.site("carrier_site").id
    # Line anchor: the carrier's rest position; axis = rope mean direction.
    self._line_pt = jp.array(
        np.asarray(self._mj_model.body("rope_carrier").pos, dtype=float)
    )
    self._slope_axis = jp.array(axis)


def register() -> None:
  locomotion.register_environment(
      "G1ClimbTerrain",
      functools.partial(G1ClimbTerrain, task="flat_terrain"),
      default_config,
  )
