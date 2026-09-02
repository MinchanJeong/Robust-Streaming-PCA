"""Synthetic streams for calibrating the drift estimator.

Built on the generator the synthetic experiments use (`src/generate_At.py`), so
that what the estimator is validated against is the same construction the rest
of the paper's experiments run on: x_t = A_t z_t + w_t with z_t ~ N(0, I_k) and
w_t ~ N(0, sigma^2 I_p) drawn independently of each other, and A_t = U_t D_t V_t^T
carrying spectral gap delta and per-step drift at most Gamma.

Calibration uses its `rotate_type=True` mode: one index pair (1, p), a fixed
angle asin(Gamma/delta), and no sign flip, so ||S_t - S_{t-1}|| sits at Gamma at
every step rather than under it.  ||Delta_l||_F is then one value throughout the
stream, which is what turns the average over window pairs into an average over
repetitions of a known target.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from generate_At import SVD_base  # noqa: E402
from numba import njit  # noqa: E402


@njit
def _seed_numba(s):
    np.random.seed(s)


def aligned_stream(gamma, p=137, T=10677, k=1, delta=1.636e-2, sigma2=2.98e-4, seed=0):
    """Stream whose per-step drift is held at `gamma` for every t.

    `gamma = 0` gives the drift-free null: same spectrum, static subspace.

    Both random streams have to be seeded: the initial orthogonal U is drawn in
    the interpreter, while every step and every observation is drawn inside the
    compiled code, which keeps its own generator state.
    """
    np.random.seed(seed)
    _seed_numba(seed)
    model = SVD_base(
        p=p, k=k, sigma=np.sqrt(sigma2), delta=delta,
        gamma=max(gamma, 0.0), N=1, rotate_type=True,
    )
    X = np.empty((p, T))
    for t in range(T):
        X[:, t] = model._generate_xt()[:, 0]
    return X
