"""The climb rhythm: SLIDE (push the ascender ~0.5 m up the rope, feet still) then
WALK (walk up to the ascender), repeat. One bit of "command" in the observation;
the runtime (training env, sim2sim, Jetson) flips it from rope progress and the
pelvis-to-ascender distance. Shared by the env and sim2sim.py.
"""

from __future__ import annotations

import torch

WALK, SLIDE = 0.0, 1.0
STROKE_M = 0.5  # rope pushed per slide phase
CATCH_UP_M = 0.30  # walk until the ascender is this close (along x) to the pelvis


def update_mode(
  mode: torch.Tensor, slide_q: torch.Tensor, slide_at_switch: torch.Tensor, rel_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return (new_mode, new_slide_at_switch).

  mode: [n] WALK/SLIDE, slide_q: [n] rope progress, slide_at_switch: [n] rope
  progress when the current phase started, rel_x: [n] ascender x minus pelvis x.
  """
  done_slide = (mode == SLIDE) & (slide_q - slide_at_switch >= STROKE_M)
  done_walk = (mode == WALK) & (rel_x <= CATCH_UP_M)
  new_mode = torch.where(done_slide, torch.full_like(mode, WALK), mode)
  new_mode = torch.where(done_walk, torch.full_like(mode, SLIDE), new_mode)
  switched = done_slide | done_walk
  return new_mode, torch.where(switched, slide_q, slide_at_switch)
