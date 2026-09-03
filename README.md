# Robust Streaming PCA

Code for **Robust Streaming PCA** (NeurIPS 2022).

[arXiv:1902.03223](https://arxiv.org/abs/1902.03223) | [OpenReview](https://openreview.net/forum?id=dAZdQM32IoK) | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/file/1b11d918b08f781a6c194c6c522edfd6-Paper-Conference.pdf)

> **The [arXiv version](https://arxiv.org/abs/1902.03223) supersedes the proceedings paper.** The 2026 revision corrects and expands the proofs, and replaces the earlier drift reading on S&P500 with a calibrated estimator ([`real_data/drift_estimation/`](real_data/drift_estimation/)).

## What the paper studies

Streaming PCA assumes the covariance the data are drawn from stands still. It does not, in general, and an algorithm tuned for a static target degrades once the target moves.

The paper replaces the static assumption with a **temporal uncertainty set**: the signal part of the covariance may move by at most `Γ` in operator norm at every step, against a spectral gap `δ` that separates the `k` directions being tracked from the rest. Two algorithms are analyzed under it, a block **noisy power method** and **Oja's algorithm**, and the resulting rates are matched against a **minimax lower bound** over the same uncertainty set.

The tension the experiments measure is a single trade-off. Averaging over a longer block suppresses sampling noise but averages over a target that has moved, so both the block size `B` (power method) and the step size `ζ` (Oja) have an interior optimum set by `Γ`, and the achievable error does not go to zero.

## Layout

| directory | what it is |
|---|---|
| [`src/`](src/) | `generate_At.py`: the generator every synthetic experiment draws from. Produces `A_t = U_t D_t V_t^T` with spectral gap `δ` and per-step drift bounded by `Γ`, and the observations `x_t = A_t z_t + w_t`. |
| [`fig_a/`](fig_a/) | Error against block size / step size, at four drift levels including `Γ = 0`. |
| [`fig_a2/`](fig_a2/) | The same sweep over a wider drift range (`Γ` up to `1e-3`), fewer repetitions. |
| [`fig_b/`](fig_b/) | Four-stage pipeline: a coarse sweep, a per-setting range selection from it, a fine sweep inside that range, and the plot. |
| [`real_data/`](real_data/) | The same two algorithms on the S&P500 daily-return stream, for `k = 1..5`. |
| [`real_data/drift_estimation/`](real_data/drift_estimation/) | Measures `Γ` and `δ` on that stream rather than assuming them. Appendix I of the 2026 revision; see the README there. |

## Reproducing

Each experiment writes result files that the notebook beside it reads, so a figure is *run the shell script, then open the notebook*.

**Figure 1-(a)**, error against `B` and `ζ` at several drift levels:

```bash
cd fig_a
source noisy.sh     # block noisy power method
source oja.sh       # Oja
```

then run `mainplot_fig1a.ipynb`. The wider drift range is `fig_a2/` with `noisy2.sh` / `oja2.sh` and `mainplot_additional.ipynb`.

**Figure 1-(b)**, the fine-tuned sweep. One script drives all four stages:

```bash
cd fig_b
source run.sh
```

**Real data.** The stream is built from the Kaggle stock-market dataset named in [`real_data/datasets/sp500/readme.txt`](real_data/datasets/sp500/readme.txt); place its `csv/` directory there, then:

```bash
cd real_data
python preprocess.py    # writes datasets/sp500.npy, shape (137, 10677)
source noisy.sh
source oja.sh
```

and run `data_analysis.ipynb` / `plotting.ipynb`.

Both `main.py` entry points take the same flags: `-algo {noisy,oja}`, `-loop <repetitions>`, plus `-gamma <Γ>` for the synthetic runs and `-k <k>` for the real-data ones.

## Requirements

`numpy`, `numba`, `matplotlib`, `scikit-learn`, `pandas`, `tqdm`, listed in [`requirements.txt`](requirements.txt). The inner loops are `numba`-compiled, so each run is slow until the first call finishes compiling.

## Citation

```bibtex
@inproceedings{robuststreamingpca,
  title     = {Robust Streaming {PCA}},
  author    = {Bienstock, Daniel and Jeong, Minchan and Shukla, Apurv and Yun, Se-Young},
  booktitle = {Advances in Neural Information Processing Systems},
  editor    = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
  pages     = {4231--4243},
  publisher = {Curran Associates, Inc.},
  volume    = {35},
  year      = {2022},
  note      = {Authors ordered alphabetically.}
}
```
