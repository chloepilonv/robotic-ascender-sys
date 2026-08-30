# Humans (for the detection gate)

`human.xml` — a 1.70 m person as capsules on one **mocap** body. No physics, no
collision: it exists to be *seen* by `app/safety/human_gate.py`, not walked into.

- Place: `data.mocap_pos[model.body_mocapid[body_id]] = (x, y, z_ground)`.
- Detect: body names `human_*`, geom group 2, geom names `human_*` (segmentation
  rendering of the `d435i` camera returns those geom ids).
- Attach to any scene: `assets/humans/humans.py::attach_humans(spec, count)`.

Standalone check: `python -c "import mujoco; mujoco.MjModel.from_xml_path('assets/humans/human.xml')"`.

The runtime harness (`app/harness/runtime.py --human <m>`) works with or without
these bodies: with them, `HumanWorld` moves the mocap bodies; without them (the
team model is compiled from mujoco_playground's XML and cannot be re-specced
from here) it falls back to virtual, draw-only humans.
