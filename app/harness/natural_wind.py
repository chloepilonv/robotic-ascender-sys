"""Natural wind: the dial's number as a TARGET a stochastic process wanders around.

Steady wind is a dial reading. Real mountain wind on a face is a mean with slow
drift and gusts on top of it, and the difference is the whole visual and audible
character of the thing -- ribbons that surge and slacken rather than lying at a
fixed angle.

The model, all of it standard and none of it invented here:

  speed  = target * gain,  gain = clamp(1 + OU(sigma 0.25, tau 4 s), 0.4, 1.6)
                                  * gust(t)
  head   = target_heading + OU(sigma 15 deg, tau 6 s)

`OU` is an Ornstein-Uhlenbeck process -- the standard "wanders around a mean and
is pulled back to it" noise. Exactly integrated per tick rather than Euler-
stepped, so the process has the right variance at any control rate:

    x <- x * exp(-dt/tau) + sigma * sqrt(1 - exp(-2 dt/tau)) * N(0, 1)

Euler would make sigma and tau depend on the tick rate, which is how "the wind
felt different at 20 Hz" bugs happen.

`gust` is a separate arrival process: every 3-8 s a gust starts, lasting
0.5-1.5 s, adding 20-60%, shaped by a raised cosine so it rises and decays
smoothly instead of stepping. A step in wind force is both visually wrong and a
discontinuity in the physics.

DETERMINISM. Everything draws from one `numpy.random.Generator` seeded from the
run's `--seed`, and the process advances exactly once per control tick, so a
replay at the same seed produces the same wind. Nothing reads the wall clock.

Inputs  : `target_speed` m/s and `target_heading_degrees`, per tick, from the
          dial; `dt` the control period.
Outputs : `.velocity_world` (2,) m/s -- the INSTANTANEOUS wind vector the
          physics is stepped with; `.speed` m/s and `.heading_degrees` for the
          page's ribbons, sound and HUD.

Switched off (`enabled=False`) it is a straight pass-through: the vector is
exactly the dial's, to the bit.
"""

import math

import numpy as np

DRIFT_SIGMA = 0.25          # fractional, 1-sigma of the slow speed drift
DRIFT_TAU_SECONDS = 4.0
DRIFT_CLAMP = (0.4, 1.6)    # gain never outside this, however the draw lands

HEADING_SIGMA_DEGREES = 15.0
HEADING_TAU_SECONDS = 6.0

GUST_INTERVAL_SECONDS = (3.0, 8.0)
GUST_DURATION_SECONDS = (0.5, 1.5)
GUST_STRENGTH = (0.20, 0.60)   # fractional boost at the peak


class OrnsteinUhlenbeck:
    """Mean-reverting noise, integrated exactly. Mean 0, stationary sd sigma."""

    def __init__(self, sigma, tau_seconds, random):
        self.sigma = float(sigma)
        self.tau = float(tau_seconds)
        self.random = random
        self.value = 0.0

    def step(self, dt: float) -> float:
        decay = math.exp(-dt / self.tau)
        self.value = (decay * self.value
                      + self.sigma * math.sqrt(max(0.0, 1.0 - decay * decay))
                      * self.random.standard_normal())
        return self.value

    def reset(self) -> None:
        self.value = 0.0


class NaturalWind:
    """The dial's target -> an instantaneous wind vector that gusts and drifts."""

    def __init__(self, seed=0, enabled=False):
        self.random = np.random.default_rng(seed)
        self.enabled = bool(enabled)
        self.drift = OrnsteinUhlenbeck(DRIFT_SIGMA, DRIFT_TAU_SECONDS, self.random)
        self.heading_drift = OrnsteinUhlenbeck(
            HEADING_SIGMA_DEGREES, HEADING_TAU_SECONDS, self.random)
        self.velocity_world = np.zeros(2)
        self.speed = 0.0
        self.heading_degrees = 0.0
        self.gain = 1.0
        self.gust_amount = 0.0
        self._reset_gust(0.0)

    def _reset_gust(self, now: float) -> None:
        self._gust_start = now + float(self.random.uniform(*GUST_INTERVAL_SECONDS))
        self._gust_duration = float(self.random.uniform(*GUST_DURATION_SECONDS))
        self._gust_strength = float(self.random.uniform(*GUST_STRENGTH))

    def reset(self) -> None:
        self.drift.reset()
        self.heading_drift.reset()
        self.gain = 1.0
        self.gust_amount = 0.0
        self._reset_gust(0.0)

    def _gust(self, now: float) -> float:
        """Raised-cosine bump, 0 outside the gust. Rolls the next one when done."""
        if now < self._gust_start:
            return 0.0
        elapsed = now - self._gust_start
        if elapsed >= self._gust_duration:
            self._reset_gust(now)
            return 0.0
        # 0 -> 1 -> 0 over the duration, with zero slope at both ends.
        return self._gust_strength * 0.5 * (
            1.0 - math.cos(2.0 * math.pi * elapsed / self._gust_duration))

    def step(self, target_velocity_world, dt: float, now: float):
        """Advance one control tick. -> the instantaneous wind vector (2,) m/s."""
        target = np.asarray(target_velocity_world, dtype=float)
        target_speed = float(np.linalg.norm(target))
        target_heading = math.degrees(math.atan2(target[1], target[0]))

        if not self.enabled or target_speed == 0.0:
            # Off, or no wind asked for: the dial value, unmodified.
            self.gain, self.gust_amount = 1.0, 0.0
            self.speed = target_speed
            self.heading_degrees = target_heading
            self.velocity_world[:] = target
            return self.velocity_world

        self.gain = float(np.clip(1.0 + self.drift.step(dt), *DRIFT_CLAMP))
        self.gust_amount = self._gust(now)
        self.speed = target_speed * self.gain * (1.0 + self.gust_amount)
        self.heading_degrees = target_heading + self.heading_drift.step(dt)

        radians = math.radians(self.heading_degrees)
        self.velocity_world[:] = (self.speed * math.cos(radians),
                                  self.speed * math.sin(radians))
        return self.velocity_world

    def report(self) -> dict:
        """The instantaneous state, for the state message and the recorder."""
        return {
            "wind_speed_mps": self.speed,
            "wind_heading_degrees": self.heading_degrees,
            "wind_gain": self.gain,
            "wind_gust": self.gust_amount,
            "wind_natural": 1.0 if self.enabled else 0.0,
        }


if __name__ == "__main__":
    # Distributional sanity: print what the process actually does, because a
    # wind model that looks plausible in code and is broken in numbers is the
    # easiest thing in the world to ship.
    import collections
    dt, target = 0.02, np.array([12.0, 0.0])
    for seed in (0, 1):
        wind = NaturalWind(seed=seed, enabled=True)
        speeds, headings, gusts = [], [], []
        for k in range(int(300 / dt)):
            wind.step(target, dt, k * dt)
            speeds.append(wind.speed); headings.append(wind.heading_degrees)
            gusts.append(wind.gust_amount)
        speeds, headings, gusts = map(np.array, (speeds, headings, gusts))
        active = gusts > 1e-6
        # count gust onsets
        onsets = int(np.sum(active[1:] & ~active[:-1]))
        print(f"seed {seed}: speed mean {speeds.mean():6.2f} (target 12.00)"
              f"  sd {speeds.std():5.2f}  min {speeds.min():5.2f}"
              f"  max {speeds.max():6.2f}")
        print(f"         heading mean {headings.mean():+6.2f} deg"
              f"  sd {headings.std():5.2f} (target sd 15.00)")
        print(f"         gusts {onsets} in 300 s"
              f" ({300 / max(onsets, 1):.1f} s apart, expect 3-8 + duration)"
              f"  in-gust fraction {active.mean():.3f}")
    a = NaturalWind(seed=5, enabled=True); b = NaturalWind(seed=5, enabled=True)
    same = all(np.allclose(a.step(target, dt, k * dt), b.step(target, dt, k * dt))
               for k in range(2000))
    print(f"determinism at equal seed: {'PASS' if same else 'FAIL'}")
    off = NaturalWind(seed=5, enabled=False)
    exact = all(np.array_equal(off.step(target, dt, k * dt), target)
                for k in range(500))
    print(f"disabled is bit-exact pass-through: {'PASS' if exact else 'FAIL'}")
