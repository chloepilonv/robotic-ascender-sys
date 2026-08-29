"""Pure-NumPy inference for the mels G1 joystick policy.

Deliberately has no JAX/brax/mujoco dependency: the training stack stays on the
workstation, only the .npz crosses to the robot. The whole network is 225k
params (0.90 MB fp32) and ~0.45 MFLOP per step -- roughly 15-50 us on one CPU
core, against a 20 ms budget. Run it on the CPU; a net this small on the GPU is
dominated by launch and sync overhead, and the jitter is worse.

All buffers are preallocated and reused so the control loop never allocates.
"""
import numpy as np

from . import constants as C


class Policy:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self._w = [z[f"hidden_{i}_kernel"].astype(np.float32) for i in range(4)]
        self._b = [z[f"hidden_{i}_bias"].astype(np.float32) for i in range(4)]
        self._mu = z["obs_mean"].astype(np.float32)
        self._sd = z["obs_std"].astype(np.float32)
        # Infer which observation view this checkpoint wants from its input
        # width, so a no-velocity policy and the 103-dim baseline can be fed
        # from the same canonical observation without a flag to get wrong.
        self.obs_dim = int(self._w[0].shape[0])
        self._view = C.view_for(self.obs_dim)
        self.uses_linvel = self._view is None
        if self._w[3].shape[1] != 2 * C.ACTION_DIM:
            raise ValueError(f"expected {2*C.ACTION_DIM} outputs "
                             f"(mean+logstd), got {self._w[3].shape[1]}")
        # Reused scratch: normalized obs, three hidden activations, output.
        self._x = np.empty(self.obs_dim, dtype=np.float32)
        self._in = np.empty(self.obs_dim, dtype=np.float32)
        self._h = [np.empty(w.shape[1], dtype=np.float32) for w in self._w]
        # Separate scratch: _swish must not alias its input (dividing a buffer
        # by itself silently yields all ones).
        self._t = [np.empty(w.shape[1], dtype=np.float32) for w in self._w]

    @staticmethod
    def _swish(x, tmp):
        # x / (1 + exp(-x)), the brax PPO default activation. `tmp` must be a
        # distinct buffer from `x`.
        np.negative(x, out=tmp); np.exp(tmp, out=tmp)
        np.add(tmp, 1.0, out=tmp); np.divide(x, tmp, out=x)
        return x

    def __call__(self, obs) -> np.ndarray:
        """Canonical 103-dim obs -> raw action (29,).

        Always takes the FULL canonical observation regardless of what this
        checkpoint was trained on; the view is applied here. Deterministic --
        the log-std half of the output is dropped.

        WARNING: the return value is a view into a reused buffer and is
        INVALIDATED by the next call. Copy it if you need to keep it -- this
        matters when running two policies side by side, or when comparing an
        action against the previous step's.
        """
        if self._view is not None:
            np.take(obs, self._view, out=self._in)
            obs = self._in
        np.subtract(obs, self._mu, out=self._x)
        np.divide(self._x, self._sd, out=self._x)
        h = self._x
        for i in range(3):
            np.dot(h, self._w[i], out=self._h[i])
            np.add(self._h[i], self._b[i], out=self._h[i])
            h = self._swish(self._h[i], self._t[i])
        np.dot(h, self._w[3], out=self._h[3])
        np.add(self._h[3], self._b[3], out=self._h[3])
        return self._h[3][:C.ACTION_DIM]

    @staticmethod
    def motor_targets(action) -> np.ndarray:
        """Raw action -> joint position targets for LowCmd."""
        return C.DEFAULT_POSE + C.ACTION_SCALE * np.asarray(action, np.float32)
