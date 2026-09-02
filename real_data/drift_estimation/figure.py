"""Plot the per-window drift reading across the four decades of the stream.

Q_hat_l^{1/2} at each window pair, before any averaging over l, on a log scale.
Pairs whose estimate is not positive carry no reading and are omitted.

Run from this directory:  python figure.py
Needs ../datasets/sp500.npy and ../datasets/sp500_dates.npy (see ../preprocess.py).
"""

import datetime
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from estimator import crossfit_q, truncated, window_pairs  # noqa: E402

DATA = os.path.join("..", "datasets", "sp500.npy")
DATES = os.path.join("..", "datasets", "sp500_dates.npy")
OUT = "sp_Gamma.pdf"
K = 1
WS = [50, 100, 200, 500, 1000]
COLOURS = ["#E03131", "#D9A400", "#2F9E44", "#1971C2", "#7048E8"]
MARKERS = ["o", "s", "^", "D", "v"]


def main():
    stream = np.load(DATA)
    dates = np.load(DATES, allow_pickle=True)
    p, T = stream.shape
    axis = np.array([datetime.date(int(s[:4]), int(s[4:6]), int(s[6:])) for s in dates])

    fig, ax = plt.subplots(figsize=(12, 9))
    report = []
    for i, w in enumerate(WS):
        pairs = window_pairs(T, w)
        q = crossfit_q(stream, w, truncated(K))
        keep = q > 0
        centres = [axis[(a[0] + b[1]) // 2] for a, b in pairs]
        ax.plot(
            [c for c, k in zip(centres, keep) if k], np.sqrt(q[keep]),
            marker=MARKERS[i], ms=5.0, lw=1.6, color=COLOURS[i], alpha=0.9,
            label=f"$w={w}$",
        )
        report.append((w, len(pairs), int((~keep).sum())))

    ax.set_yscale("log")
    ax.set_xlabel("Date", fontsize=28)
    ax.set_ylabel(r"$\widehat{Q}_\ell^{\,1/2}$", fontsize=30)
    ax.tick_params(labelsize=23)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=23, ncol=2, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    print(f"{'w':>6} {'pairs':>6} {'unresolved':>11}")
    for w, n, nneg in report:
        print(f"{w:6d} {n:6d} {nneg:11d}")


if __name__ == "__main__":
    main()
