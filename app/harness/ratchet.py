"""The ascender ratchet, in plain MuJoCo, with THEIR semantics verbatim.

Their version is JAX array ops inside `jax.lax.scan` over substeps
(`rl/environment/climb_env.py:268-285`, `_step_physics`):

    prev_slide = data.qpos[slide_q]        # BEFORE the substep
    data = mjx.step(...)
    qvel = qvel.at[slide_dof].set(max(qvel[slide_dof], 0.0))
    qpos = qpos.at[slide_q].set(max(qpos[slide_q], prev_slide))

So: the hand slides UP the line freely and can never slide back DOWN, and the
clamp fires once per PHYSICS substep (500 Hz), not once per control tick.
Same-order-of-operations matters -- `prev_slide` is read before the step, and
the clamp is applied after it, on the integrated state, with no forward call in
between (so `data.sensordata` / `data.xpos` stay one substep stale exactly as
they do on their side).

Inputs
------
data.qpos[slide_qpos_address] : ascender travel along the line, metres,
    0 at the reset grip point, positive uphill.
data.qvel[slide_dof_address]  : ascender rate along the line, m/s.

Outputs
-------
In-place mutation of `data.qpos` / `data.qvel`. `travel_meters` reads the
clamped coordinate back out.
"""

import numpy as np


class AscenderRatchet:
    """Per-substep unidirectional clamp on the `ascender_slide` coordinate."""

    def __init__(self, slide_qpos_address: int, slide_dof_address: int):
        self.slide_qpos_address = int(slide_qpos_address)
        self.slide_dof_address = int(slide_dof_address)
        self.previous_travel_meters = 0.0
        self.highest_travel_meters = 0.0

    def before_substep(self, data) -> None:
        """Latch `prev_slide` -- climb_env.py:274."""
        self.previous_travel_meters = float(data.qpos[self.slide_qpos_address])

    def after_substep(self, data) -> None:
        """Clamp rate non-negative and travel non-decreasing -- climb_env.py:277-282."""
        if data.qvel[self.slide_dof_address] < 0.0:
            data.qvel[self.slide_dof_address] = 0.0
        if data.qpos[self.slide_qpos_address] < self.previous_travel_meters:
            data.qpos[self.slide_qpos_address] = self.previous_travel_meters
        self.highest_travel_meters = max(
            self.highest_travel_meters, float(data.qpos[self.slide_qpos_address])
        )

    def reset(self, data) -> None:
        self.previous_travel_meters = float(data.qpos[self.slide_qpos_address])
        self.highest_travel_meters = self.previous_travel_meters

    def travel_meters(self, data) -> float:
        """Ascender travel along the line, metres from the reset grip point."""
        return float(data.qpos[self.slide_qpos_address])

    def travel_rate_meters_per_second(self, data) -> float:
        return float(data.qvel[self.slide_dof_address])


def step_with_ratchet(mujoco_module, model, data, ratchet, substeps: int) -> None:
    """One control step: `substeps` x (mj_step + ratchet). Their `_step_physics`.

    `data.ctrl` must already hold the motor targets; it is NOT written here
    (their scan re-sets the same ctrl every substep, which is equivalent).
    No `mj_forward` afterwards -- see the module docstring on staleness.
    """
    for _ in range(substeps):
        ratchet.before_substep(data)
        mujoco_module.mj_step(model, data)
        ratchet.after_substep(data)


def hand_height_on_line_meters(data, palm_site_id, line_point_world,
                               slope_axis_world) -> float:
    """Right-palm arc length up the line -- climb_env.py:544-547."""
    palm = np.asarray(data.site_xpos[palm_site_id])
    return float((palm - np.asarray(line_point_world)) @ np.asarray(slope_axis_world))


def hand_line_error_meters(data, palm_site_id, line_point_world,
                           slope_axis_world) -> float:
    """Right-palm perpendicular distance off the line -- climb_env.py:536-542."""
    axis = np.asarray(slope_axis_world)
    relative = np.asarray(data.site_xpos[palm_site_id]) - np.asarray(line_point_world)
    return float(np.linalg.norm(relative - (relative @ axis) * axis))
