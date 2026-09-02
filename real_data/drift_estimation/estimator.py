"""Cross-fit estimator for the signal drift of a covariance stream.

The plug-in reading ||Sigma_hat_{l+1} - Sigma_hat_l||_F^2 carries a sampling
floor of order tr(Sigma)^2 / w, which on this data is larger than the drift it
is meant to measure.  The estimator here removes that floor by pairing two
estimates instead of squaring one: the observations of a window are split by
the parity of their offset, each half gives its own window difference, and the
two are combined in a Frobenius inner product.  The halves are independent, so
the product of their sampling errors has mean zero and

    E Q_hat_l = <E Delta_hat^(1)_l, E Delta_hat^(2)_l>_F,

which the paper's Appendix I relates to ||Delta_l||_F^2 up to a bias that is
subtracted rather than added.

Q_hat_l is an unbiased-up-to-that-bias estimate of a squared Frobenius norm, so
it can come out negative on a pair where the drift is below the noise; such
pairs are reported as unresolved rather than rectified to zero.
"""

import numpy as np


def window_pairs(T, w):
    """Consecutive disjoint length-w window pairs, right-aligned on the stream."""
    idx = [(T % w + i * w, T % w + (i + 1) * w) for i in range(T // w)]
    return list(zip(idx[:-1], idx[1:]))


def sample_covariance(X):
    """Second moment of the columns of X."""
    return X @ X.T / X.shape[1]


def signal_part(X, k):
    """Rank-k signal estimate: sum_{i<=k} (lam_i - sigma2_hat) v_i v_i^T.

    sigma2_hat is the mean of the trailing p-k eigenvalues, so the isotropic
    noise level is removed and what remains estimates A_t A_t^T rather than
    Sigma_t.  The truncation is what lets the window average stand in for a
    single Sigma_t: under the paper's assumption the covariances of one window
    share a k-dimensional subspace, so keeping k eigenpairs recovers their
    average instead of cutting part of it away.
    """
    S = sample_covariance(X)
    lam, V = np.linalg.eigh(S)
    lam, V = lam[::-1], V[:, ::-1]
    sigma2 = lam[k:].mean()
    return (V[:, :k] * np.maximum(lam[:k] - sigma2, 0.0)) @ V[:, :k].T


def truncated(k):
    """Projection used for the top-k estimator."""
    return lambda X: signal_part(X, k)


def crossfit_q(stream, w, project):
    """Q_hat_l for every consecutive length-w window pair.

    `project` maps a block of observations to the matrix whose drift is being
    measured -- `truncated(k)` for the top-k signal, `sample_covariance` for the
    full second moment.
    """
    T = stream.shape[1]
    out = []
    for (a0, a1), (b0, b1) in window_pairs(T, w):
        A, B = stream[:, a0:a1], stream[:, b0:b1]
        d1 = project(A[:, 0::2]) - project(B[:, 0::2])
        d2 = project(A[:, 1::2]) - project(B[:, 1::2])
        out.append(float(np.sum(d1 * d2)))
    return np.array(out)


def permutation_null(stream, w, project, nperm, rng):
    """Q_hat_l with the two windows' observations pooled and reshuffled.

    Destroys the ordering that carries the drift while leaving the sampling
    behaviour of the estimator intact, so the average over permutations is what
    Q_hat_l reads on a stream with the same spectrum and no drift.
    """
    T = stream.shape[1]
    out = []
    for (a0, a1), (b0, b1) in window_pairs(T, w):
        pool = np.concatenate([stream[:, a0:a1], stream[:, b0:b1]], axis=1)
        acc = []
        for _ in range(nperm):
            q = pool[:, rng.permutation(2 * w)]
            A0, B0 = q[:, :w], q[:, w:]
            e1 = project(A0[:, 0::2]) - project(B0[:, 0::2])
            e2 = project(A0[:, 1::2]) - project(B0[:, 1::2])
            acc.append(float(np.sum(e1 * e2)))
        out.append(np.mean(acc))
    return np.array(out)


def resolve(q):
    """sqrt of a mean Q_hat, or 0.0 where the mean is not positive."""
    return float(np.sqrt(max(np.mean(q), 0.0)))


def bootstrap_ci(q, rng, nboot=1500, alpha=0.05):
    """Percentile interval for resolve(q) over resampled window pairs."""
    m = len(q)
    draws = [resolve(q[rng.integers(0, m, m)]) for _ in range(nboot)]
    return tuple(np.quantile(draws, [alpha / 2, 1 - alpha / 2]))


def spectral_gap(stream, w, kmax):
    """delta_hat = s_k(Sigma_hat_l) - sigma2_hat, minimised over length-w windows.

    Definition 1 asks for the gap to hold at every t, so the reported figure is
    the minimum over windows rather than the average.
    """
    T = stream.shape[1]
    idx = [(T % w + i * w, T % w + (i + 1) * w) for i in range(T // w)]
    lams = [np.linalg.eigvalsh(sample_covariance(stream[:, a:b]))[::-1] for a, b in idx]
    return {k: min(l[k - 1] - l[k:].mean() for l in lams) for k in range(1, kmax + 1)}
