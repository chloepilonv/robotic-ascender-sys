"""Put humans from human.xml into ANY MjSpec before it compiles.

    spec = mujoco.MjSpec.from_file("assets/robots/mujoco/g1_unitree.xml")
    attach_humans(spec, count=3)          # bodies human_0, human_1, human_2
    model = spec.compile()

Each copy is a mocap body: place it with `data.mocap_pos[mocap_id]`, where
`mocap_id = model.body_mocapid[body_id]`. Bodies start parked at PARK_POSITION
(far below the terrain) so an unspawned human is never in view.

Nothing here touches the robot or the terrain; humans have no collision.
"""
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HUMAN_XML = os.path.join(HERE, "human.xml")
BODY_PREFIX = "human_"
PARK_POSITION = np.array([0.0, 0.0, -100.0])


def attach_humans(spec: mujoco.MjSpec, count: int = 1) -> list:
    """Attach `count` copies; returns the attached body names."""
    names = []
    for index in range(count):
        human = mujoco.MjSpec.from_file(HUMAN_XML)
        prefix = f"{BODY_PREFIX}{index}_"
        frame = spec.worldbody.add_frame(pos=PARK_POSITION)
        frame.attach_body(human.body("human"), prefix, "")
        names.append(f"{prefix}human")
    return names


def human_body_ids(model: mujoco.MjModel) -> list:
    """Body ids of every attached human (mocap bodies named human_*)."""
    return [i for i in range(model.nbody)
            if model.body(i).name.startswith(BODY_PREFIX) and model.body_mocapid[i] >= 0]


def build_scene_with_humans(robot_xml, count=1):
    """Convenience: (model, body_ids) for a robot MJCF plus `count` humans."""
    spec = mujoco.MjSpec.from_file(robot_xml)
    attach_humans(spec, count)
    model = spec.compile()
    return model, human_body_ids(model)
