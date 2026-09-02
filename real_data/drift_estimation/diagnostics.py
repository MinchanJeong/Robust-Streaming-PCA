"""The readings behind the sentences, as against the table and the figure.

`measure.py` reproduces the reported table and `figure.py` the reported figure.
Three claims in the text are read off the same stream without appearing in
either, and this script produces them:

  1. why the cross-fit estimator is used at all, by measuring the sampling floor
     the plug-in reading carries and the cross-fit one does not;
  2. where delta_hat sits inside the spread of windows it is minimized over,
     which is what makes the minimum a conservative choice rather than an
     outlier;
  3. how unevenly the drift is spread over the four decades, and which windows
     the largest readings fall in.

A fourth block reports how much of the stream each window length resolves,
which is what sets the shortest window the figure plots.

Run from this directory:  python diagnostics.py
Needs ../datasets/sp500.npy and ../datasets/sp500_dates.npy (see ../preprocess.py).
"""

import os

import numpy as np

from estimator import (crossfit_q, permutation_null, plugin_sq, plugin_stat,
                       resolve, spectral_gaps, truncated, window_pairs)

DATA = os.path.join("..", "datasets", "sp500.npy")
DATES = os.path.join("..", "datasets", "sp500_dates.npy")
K = 1
KS = [1, 2, 3, 4, 5]
WS = [20, 50, 100, 200, 500, 1000]
W_READ = 200
W_CONCENTRATION = 100
NPERM = 4


def sampling_floor(stream, rng, project):
    """The plug-in reading, the part of it that is sampling error, and the cross-fit.

    The floor is estimated by the same permutation null the estimator uses: pool
    the two windows and reshuffle, which destroys the drift and leaves the
    sampling behaviour intact, so what the plug-in still reads is its own error.
    """
    print("1. SAMPLING FLOOR.  what the plug-in reading carries and the cross-fit does not")
    print(f"{'w':>6} {'plug-in':>10} {'floor':>10} {'floor share':>12} {'cross-fit':>10} "
          f"{'plug-in over':>13}")
    for w in WS:
        pl = plugin_sq(stream, w, project).mean()
        fl = permutation_null(stream, w, project, NPERM, rng, plugin_stat).mean()
        cf = resolve(crossfit_q(stream, w, project))
        print(f"{w:6d} {np.sqrt(pl):10.4f} {np.sqrt(fl):10.4f} {fl / pl:12.2f} "
              f"{cf:10.4f} {np.sqrt(pl) / cf:13.2f}x")
    print("\n   'floor share' is the fraction of the plug-in's squared reading that survives\n"
          "   the reshuffle, and 'plug-in over' how many times the drift it reports exceeds\n"
          "   the cross-fit one. Both fall as w grows, since the floor is the term that\n"
          "   averages away.\n")


def gap_spread(stream):
    """Where the reported delta_hat sits among the windows it is minimized over."""
    print(f"2. SPECTRAL GAP.  delta_hat over the {len(window_pairs(stream.shape[1], W_READ)) + 1} "
          f"windows of length {W_READ}")
    print(f"{'k':>3} {'min':>11} {'25%':>11} {'median':>11} {'max':>11} {'min/median':>12}")
    gaps = spectral_gaps(stream, W_READ, max(KS))
    for k in KS:
        g = gaps[k]
        lo, q1, med, hi = g.min(), np.quantile(g, 0.25), np.median(g), g.max()
        print(f"{k:3d} {lo:11.3e} {q1:11.3e} {med:11.3e} {hi:11.3e} {lo / med:12.2f}")
    print("\n   The reported figure is the minimum, since the gap is asked for at every t.\n"
          "   Its ratio to the median says how much that costs.\n")


def concentration(stream, dates, project):
    """How far the largest per-window readings stand above the rest, and when."""
    T = stream.shape[1]
    q = crossfit_q(stream, W_CONCENTRATION, project)
    pairs = window_pairs(T, W_CONCENTRATION)
    keep = q > 0
    vals = np.sqrt(q[keep])
    med = np.median(vals)
    order = np.argsort(vals)[::-1]

    def span(pair):
        (a0, _), (_, b1) = pair
        return dates[a0], dates[(a0 + b1) // 2], dates[b1 - 1]

    spans = [span(pr) for pr, k in zip(pairs, keep) if k]
    print(f"3. CONCENTRATION.  largest readings at w = {W_CONCENTRATION}, against the median "
          f"{med:.4f}")
    print(f"{'rank':>5} {'Q^1/2':>10} {'over median':>12}  {'centre':>10}  covering")
    for r, i in enumerate(order[:6], start=1):
        start, mid, end = (f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in spans[i])
        print(f"{r:5d} {vals[i]:10.4f} {vals[i] / med:11.1f}x  {mid:>10}  {start} to {end}")
    print("\n   A pair is placed at its centre, which is where the figure plots it, and the\n"
          "   reading belongs to the whole 2w-day span rather than to either endpoint.\n")


def resolution(stream, project):
    """How much of the stream each window length resolves."""
    T, p = stream.shape[1], stream.shape[0]
    print("4. RESOLUTION.  window pairs whose reading is not positive carry no value")
    print(f"{'w':>6} {'pairs':>6} {'unresolved':>11} {'share':>7} {'p/w^2':>9}")
    for w in WS:
        q = crossfit_q(stream, w, project)
        n = len(q)
        bad = int((q <= 0).sum())
        print(f"{w:6d} {n:6d} {bad:11d} {bad / n:7.2f} {p / w ** 2:9.4f}")
    print("\n   p/w^2 is the reciprocal-effective-rank scale at which the cross-fit bias\n"
          "   overtakes the plug-in one, so it is small exactly where the estimator is used.\n")


def main():
    stream = np.load(DATA)
    dates = np.load(DATES, allow_pickle=True)
    rng = np.random.default_rng(101)
    project = truncated(K)
    print(f"stream: p = {stream.shape[0]}, T = {stream.shape[1]}, k = {K}\n")
    sampling_floor(stream, rng, project)
    gap_spread(stream)
    concentration(stream, dates, project)
    resolution(stream, project)


if __name__ == "__main__":
    main()
