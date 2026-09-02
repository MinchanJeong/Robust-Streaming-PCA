"""Separate the drift into the part that turns the signal subspace and the part
that only moves its eigenvalue.

Definition 1 bounds ||S_t - S_{t-1}||, which both contribute to, but only the
first is the rotation the algorithm has to track.  For k = 1 the split reads
directly: the principal angle between the leading eigenvectors of two windows
gives the rotation, and the relative change in the leading eigenvalue gives the
rest.  The angle a pair of windows shows includes what sampling noise alone
would produce, so it is reported against a permutation null taken on the same
pair.

Run from this directory:  python decompose.py
Needs ../datasets/sp500.npy (see ../preprocess.py).
"""

import math
import os

import numpy as np

from estimator import sample_covariance, window_pairs

DATA = os.path.join("..", "datasets", "sp500.npy")
WS = [100, 200, 500, 1000]
NPERM = 4


def top1(X):
    lam, V = np.linalg.eigh(sample_covariance(X))
    return lam[-1], V[:, -1]


def main():
    stream = np.load(DATA)
    T = stream.shape[1]
    rng = np.random.default_rng(7)
    print("k = 1.  sin(theta) is the principal angle between the leading "
          "eigenvectors of\nconsecutive windows; the null is the same reading "
          "after pooling and reshuffling.\n")
    print(f"{'w':>6} {'sin(theta)':>11} {'null':>9} {'excess':>9} | "
          f"{'|d lambda|':>11} {'mean lambda':>12} {'relative':>9}")
    for w in WS:
        obs, null, dlam, mlam = [], [], [], []
        for (a0, a1), (b0, b1) in window_pairs(T, w):
            A, B = stream[:, a0:a1], stream[:, b0:b1]
            la, va = top1(A)
            lb, vb = top1(B)
            obs.append(math.sqrt(max(1 - (va @ vb) ** 2, 0)))
            dlam.append(abs(la - lb))
            mlam.append((la + lb) / 2)
            pool = np.concatenate([A, B], axis=1)
            for _ in range(NPERM):
                q = pool[:, rng.permutation(2 * w)]
                _, x = top1(q[:, :w])
                _, y = top1(q[:, w:])
                null.append(math.sqrt(max(1 - (x @ y) ** 2, 0)))
        s, n = np.mean(obs), np.mean(null)
        print(f"{w:6d} {s:11.4f} {n:9.4f} {s - n:9.4f} | "
              f"{np.mean(dlam):11.5f} {np.mean(mlam):12.5f} "
              f"{np.mean(dlam) / np.mean(mlam):9.3f}")


if __name__ == "__main__":
    main()
