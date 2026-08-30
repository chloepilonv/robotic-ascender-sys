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
SLIDE_TIMEOUT_S = 3.0  # a slide phase that has not moved 0.5 m by then ends anyway (no standing forever)


def update_mode(
  mode: torch.Tensor,
  slide_q: torch.Tensor,
  slide_at_switch: torch.Tensor,
  rel_x: torch.Tensor,
  phase_t: torch.Tensor | None = None,
  dt: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return (new_mode, new_slide_at_switch, new_phase_t).

  mode: [n] WALK/SLIDE, slide_q: [n] rope progress, slide_at_switch: [n] rope
  progress when the current phase started, rel_x: [n] ascender x minus pelvis x,
  phase_t: [n] seconds spent in the current phase.
  """
  if phase_t is None:
    phase_t = torch.zeros_like(mode)
  phase_t = phase_t + dt
  done_slide = (mode == SLIDE) & ((slide_q - slide_at_switch >= STROKE_M) | (phase_t >= SLIDE_TIMEOUT_S))
  done_walk = (mode == WALK) & (rel_x <= CATCH_UP_M)
  new_mode = torch.where(done_slide, torch.full_like(mode, WALK), mode)
  new_mode = torch.where(done_walk, torch.full_like(mode, SLIDE), new_mode)
  switched = done_slide | done_walk
  return new_mode, torch.where(switched, slide_q, slide_at_switch), torch.where(switched, torch.zeros_like(phase_t), phase_t)
