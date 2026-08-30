"""Ascender telemetry from plain MuJoCo (no mjlab dependency; used by app/bms/sim)."""
def rope_state(model, data, prefix: str = "", slide_joint: str = "rope_slide") -> dict:
  """Ascender telemetry from a plain-MuJoCo (model, data) — same keys as the real
  end-effector (assets/ascender/ELECTRONICS.md): progress along the rope (m), rope
  tension (N, constraint force on the slide dof) and cam engaged (holding load)."""
  jid = model.joint(prefix + slide_joint).id
  dof = int(model.jnt_dofadr[jid])
  tension = float(abs(data.qfrc_constraint[dof]))
  return {
    "rope_progress_m": float(data.qpos[model.jnt_qposadr[jid]]),
    "tension_N": tension,
    "engaged": bool(tension > 5.0),
  }
