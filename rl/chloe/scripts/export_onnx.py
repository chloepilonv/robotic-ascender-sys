"""Export a trained rsl_rl checkpoint to ONNX for the real G1 (or any runtime).

    python -m rl.chloe.scripts.export_onnx Himalayas-Ascender-Slope20-G1 \
        logs/rsl_rl/g1_ascender_slope20/<run>/model_5000.pt policy.onnx

Input  "obs"    : float32 [1, obs_dim]  (actor observation vector, raw — the
                   exported graph contains the observation normaliser)
Output "action" : float32 [1, 29]       (joint position offsets, apply
                   `default_joint_pos + action * G1_ACTION_SCALE`, 50 Hz)
Observation order = env_cfg actor terms: base_ang_vel(3), projected_gravity(3),
joint_pos(29), joint_vel(29), last_action(29), ascender_pos_b(3).
"""

import sys
from dataclasses import asdict

import torch

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import rl.chloe.task as ascender


class _Actor(torch.nn.Module):
  def __init__(self, policy):
    super().__init__()
    self.policy = policy

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    from tensordict import TensorDict

    td = TensorDict({"actor": obs}, batch_size=[obs.shape[0]])
    return self.policy(td)


def main(task_id: str, checkpoint: str, out: str = "policy.onnx") -> None:
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env = ascender.RatchetEnv(cfg=env_cfg, device="cpu")
  wrapped = RslRlVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(wrapped, asdict(load_rl_cfg(task_id)), device="cpu")
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
  policy = runner.get_inference_policy(device="cpu")
  obs = wrapped.get_observations()["actor"].detach().clone()
  actor = _Actor(policy).eval()
  torch.onnx.export(
    actor,
    (obs,),
    out,
    input_names=["obs"],
    output_names=["action"],
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    opset_version=17,
    dynamo=False,  # legacy exporter: one self-contained file, no external weights
  )
  print(f"wrote {out}  obs_dim={obs.shape[1]}  act_dim={actor(obs).shape[1]}")
  env.close()


if __name__ == "__main__":
  main(*sys.argv[1:])
