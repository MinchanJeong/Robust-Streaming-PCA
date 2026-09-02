# Drift estimation on the S&P500 stream

Reads the two constants of Definition 1 off the S&P500 daily-return stream: the
per-step drift budget `Gamma` and the spectral gap `delta`, and the relation
between them the paper's standing hypothesis asks for. This is the code behind
Appendix I.

The quantity being measured is small. The obvious reading,
`||Sigma_hat_{l+1} - Sigma_hat_l||_F^2` between consecutive window covariances,
is dominated by its own sampling error at every window length the stream allows,
so it measures the size of `Sigma_t` rather than the distance it moved. The
estimator here removes that floor by splitting each window's observations by
the parity of their offset, forming a window difference from each half, and
taking the Frobenius inner product of the two. The halves are independent, so
the product of their sampling errors averages to zero.

Two consequences are worth knowing before reading the output:

- `Q_hat_l` estimates a squared norm without being a square, so it can come out
  negative where the drift sits below the noise. Such window pairs are reported
  as unresolved rather than rectified to zero, since rectifying them would bias
  the average upward.
- It measures the Frobenius norm of the window difference, while Definition 1
  is stated in the operator norm. Since `||.||_op <= ||.||_F`, the reading
  over-states the drift the definition constrains.

## Data

`../preprocess.py` builds `../datasets/sp500.npy`, shape `(137, 10677)`, from
the Kaggle stock-market CSVs named in `../datasets/sp500/readme.txt`, together
with `../datasets/sp500_dates.npy` for the time axis of the figure. Run it from
`real_data/` with the CSVs in place; everything here then runs from this
directory.

## Files

| file | what it does |
|---|---|
| `estimator.py` | the estimator itself: window pairing, rank-`k` signal truncation, the cross-fit inner product, the permutation null, the bootstrap interval, and the spectral gap |
| `calibrate.py` | validates it on synthetic streams whose drift is planted, before it is read on data |
| `measure.py` | the reading on the S&P500 stream: the drift budget per window length, the spectral gap per `k`, and their ratio |
| `figure.py` | `Q_hat_l^{1/2}` at each window pair across the four decades of the stream |
| `decompose.py` | splits the drift into the rotation of the signal subspace and the movement of its eigenvalue |

```
python calibrate.py     # a few minutes; generates 24 synthetic streams
python measure.py
python figure.py        # writes sp_Gamma.pdf
python decompose.py
```

## Calibration

`calibrate.py` plants a known drift and asks whether the estimator returns it.
The streams come from `src/generate_At.py`, the generator the synthetic
experiments use, run at this dataset's dimension, length and spectral scale
(`p = 137`, `T = 10677`, `k = 1`, `delta = 1.636e-2`, `sigma^2 = 2.98e-4`) in
its `rotate_type=True` mode. That mode rotates a single fixed index pair by a
fixed maximal angle, which holds `||S_t - S_{t-1}||` at `Gamma` at every step
instead of leaving it below; `||Delta_l||_F` is then one value throughout the
stream, so averaging over window pairs averages over repetitions of a known
target.

Every figure is reported over eight seeds. One stream is not enough to read
this: at `w = 100` the ratio between the estimate and the planted value spans
zero to above 1.6 across seeds, which is why `w = 100` is treated as
unresolved and the constants are read at `w = 200`.
