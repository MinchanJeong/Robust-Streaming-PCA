"""Read the drift budget and the spectral gap off the S&P500 stream.

Reproduces the table of Appendix I: Q_hat_l^{1/2} averaged over consecutive
length-w window pairs with the implied budget Gamma_hat(w) = Q_hat_l^{1/2} / w,
and delta_hat per k minimized over the length-200 windows the budget is read on.

Both constants are read at the same window length, w = 200: Definition 1 gives
Gamma >= ||Delta_l||_op / w at every w, so each window length yields a lower
bound and the largest usable one is wanted, which is the shortest w the
calibration resolves.

Run from this directory:  python measure.py
Needs ../datasets/sp500.npy (see ../preprocess.py).
"""

import os

import numpy as np

from estimator import (bootstrap_ci, crossfit_q, permutation_null, resolve,
                       spectral_gaps, truncated)

DATA = os.path.join("..", "datasets", "sp500.npy")
WS = [100, 200, 500, 1000]
KS = [1, 2, 3, 4, 5]
W_READ = 200
NPERM = 8


def main():
    stream = np.load(DATA)
    p, T = stream.shape
    rng = np.random.default_rng(101)
    top1 = truncated(1)
    print(f"stream: p = {p}, T = {T}\n")

    print("DRIFT BUDGET (k = 1)")
    print(f"{'w':>6} {'pairs':>6} {'unres.':>7} {'Q^1/2':>9} {'95% CI':>21} "
          f"{'Gamma_hat(w)':>13} {'perm. null':>11}")
    budget = {}
    for w in WS:
        q = crossfit_q(stream, w, top1)
        est = resolve(q)
        lo, hi = bootstrap_ci(q, rng)
        nul = permutation_null(stream, w, top1, NPERM, rng)
        budget[w] = est / w
        print(f"{w:6d} {len(q):6d} {int((q <= 0).sum()):7d} {est:9.4f} "
              f"[{lo:9.4f},{hi:9.4f}] {est/w:13.2e} {resolve(nul):11.2e}")

    print(f"\nSPECTRAL GAP (minimum over length-{W_READ} windows; median for scale)")
    gaps = spectral_gaps(stream, W_READ, max(KS))
    print(f"{'k':>3} {'delta_hat':>11} {'median':>11} {'Gamma_hat(200)/delta_hat':>25} {'Def. 1':>9}")
    for k in KS:
        lo = float(gaps[k].min())
        r = budget[W_READ] / lo
        print(f"{k:3d} {lo:11.2e} {np.median(gaps[k]):11.2e} {r:25.2f} "
              f"{'ok' if r < 1 else 'VIOLATED':>9}")


if __name__ == "__main__":
    main()
