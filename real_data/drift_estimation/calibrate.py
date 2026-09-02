"""Validate the drift estimator on synthetic streams before reading it on data.

Three checks, all on streams whose drift is known because it was planted:

  1. planted Gamma -- does Q_hat_l^{1/2} recover sqrt(2) * w * Gamma?
  2. planted zero  -- what does the estimator read when there is no drift?
  3. w-dependence  -- Gamma_hat(w) = Q_hat_l^{1/2} / w inverts the window bound
     exactly when the increments inside a window are aligned, so on an
     aligned-increment stream it should show no trend in w.  A trend in w read
     on the data is therefore a property of the data, not of the estimator.

Each figure is reported over several seeds.  A single stream is not enough: at
w = 100 the ratio spans zero to above 1.6 across seeds.

Run from this directory:  python calibrate.py
Check 3 additionally needs ../datasets/sp500.npy (see ../preprocess.py).
"""

import math
import os

import numpy as np

from estimator import crossfit_q, resolve, sample_covariance, truncated
from generator import aligned_stream

P, T, K = 137, 10677, 1
DELTA, SIGMA2 = 1.636e-2, 2.98e-4
WS = [100, 200, 500, 1000]
SEEDS = list(range(8))
DATA = os.path.join("..", "datasets", "sp500.npy")


def ratios(gamma, ws, seeds, project):
    """Q_hat^{1/2} divided by its planted value sqrt(2) * w * Gamma, per seed."""
    out = {w: [] for w in ws}
    for s in seeds:
        X = aligned_stream(gamma, p=P, T=T, k=K, delta=DELTA, sigma2=SIGMA2, seed=1000 + s)
        for w in ws:
            out[w].append(resolve(crossfit_q(X, w, project)) / (math.sqrt(2) * w * gamma))
    return out


def main():
    top1 = truncated(K)
    print(f"p = {P}, T = {T}, k = {K}, delta = {DELTA:.3e}, sigma^2 = {SIGMA2:.2e}, "
          f"{len(SEEDS)} seeds\n")

    print("1. PLANTED DRIFT.  estimate / sqrt(2) w Gamma, at Gamma = 1e-5")
    print(f"{'w':>6} {'mean':>7} {'min':>7} {'max':>7}   estimator")
    for label, project in (("top-1 truncated", top1), ("full covariance", sample_covariance)):
        r = ratios(1e-5, WS, SEEDS, project)
        for w in WS:
            a = np.array(r[w])
            print(f"{w:6d} {a.mean():7.2f} {a.min():7.2f} {a.max():7.2f}   {label}")
        print()

    print("2. ZERO-DRIFT NULL.  Q_hat^{1/2} at Gamma = 0, against the data reading")
    data_reading = {}
    if os.path.exists(DATA):
        stream = np.load(DATA)
        for w in WS:
            data_reading[w] = resolve(crossfit_q(stream, w, top1))
    print(f"{'w':>6} {'unresolved':>11} {'max over seeds':>15} {'data':>9} {'ratio':>7}")
    for w in WS:
        vals = []
        for s in SEEDS:
            X = aligned_stream(0.0, p=P, T=T, k=K, delta=DELTA, sigma2=SIGMA2, seed=2000 + s)
            vals.append(resolve(crossfit_q(X, w, top1)))
        vals = np.array(vals)
        nzero = int((vals == 0.0).sum())
        d = data_reading.get(w)
        tail = f"{d:9.4f} {vals.max()/d:7.3f}" if d else f"{'--':>9} {'--':>7}"
        print(f"{w:6d} {nzero:>4d}/{len(SEEDS):<6d} {vals.max():15.4e} {tail}")
    print()

    if not data_reading:
        print(f"3. skipped: {DATA} not found.")
        return
    print("3. W-DEPENDENCE.  Gamma_hat(w) = Q_hat^{1/2} / w, aligned stream vs data")
    print(f"{'w':>6} | {'synthetic':>11} {'/w=100':>7} | {'data':>11} {'/w=100':>7}")
    syn = aligned_stream(1e-5, p=P, T=T, k=K, delta=DELTA, sigma2=SIGMA2, seed=11)
    sy = {w: resolve(crossfit_q(syn, w, top1)) / w for w in WS}
    da = {w: data_reading[w] / w for w in WS}
    for w in WS:
        print(f"{w:6d} | {sy[w]:11.3e} {sy[w]/sy[100]:7.2f} | {da[w]:11.3e} {da[w]/da[100]:7.2f}")


if __name__ == "__main__":
    main()
