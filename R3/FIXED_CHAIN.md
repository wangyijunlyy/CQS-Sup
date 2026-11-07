# Fixed-data chain comparison

This extension supports W3.3 of the R3 response. It is separate from the additional-measurement local-correction experiments.

## Setting and interpretation

The task is six-qubit, two-layer TFIM VQE on a one-dimensional chain, with 16 effective controls. Each of ten fresh independent datasets has 2,000 training controls, 400 measured-validation controls and 400 held-out controls. Ten random-Pauli measurements at each training/validation control give 24,000 simulated acquisition shots per dataset. Search uses no additional measurements. All six couplings and all 20 search outputs are retained.

CQS-P uses moment-supervised training with ranks 2/4/6/8/16/32 and five regularization settings, plus the original rank-32 recipe. Measured validation-energy MSE across the six couplings selects configurations and the new-fit checkpoints. Complete Fourier uses the same masks and frequency coverage. The released baselines include moment-validation ridge selection and an expanded regularization family selected by the same energy-validation criterion. The protocol is frozen before the ten new datasets are generated. Exact energies score saved outputs only.

Each search uses 20 identical paired starts, 100 Adam updates, learning rate 0.1 and Gaussian continuation from 1 to 0 over 60 steps. Uniform starts retain angle clipping. The reference-neighborhood starts perturb all 16 active coordinates by uniform ±0.2 and use periodic wrapping. Results first average all outputs and six couplings within each dataset, then summarize the ten independent dataset means.

| Method | Uniform starts | Reference-neighborhood starts |
| --- | ---: | ---: |
| CQS-P+C | −2.228 ± 0.243 | −2.499 ± 0.362 |
| Complete Fourier+C, moment-validation selected | −2.040 ± 0.094 | −2.112 ± 0.179 |
| Complete Fourier+C, energy-validation selected | −2.050 ± 0.100 | −2.118 ± 0.182 |

CQS-P wins against either baseline on 7/10 and 9/10 datasets in the respective settings. The common uniform-start initial energy is 0.005. The reference-neighborhood initial mean is −2.934, below the terminal means, so that comparison supports relative terminal quality, not improvement over its initialization or attainment below −3. The uniform-start paired interval crosses zero; the reference-neighborhood paired intervals favor CQS-P. Full numerical summaries retain all intervals and per-coupling results. These are matched-step comparisons, not equal-wall-time claims.

## Replay the released evidence

From R3, after installing requirements:

```bash
python fetch_artifacts.py
python reproduce.py --suite chain-fixed
```

The added release asset contains all ten raw measurement records and datasets, all 310 product model checkpoints, the complete Fourier coefficients and validation candidates, identical starts, all 360 saved searches, and independent statevector scoring records. The replay reconstructs measured labels, checks every selected checkpoint and Fourier baseline, checks paired starts and continuation schedules, and recalculates all true terminal energies. Expected: ten complete datasets, 310 product fits, 360 searches, and zero additional search shots. It writes summaries and verification records within the extracted experiment directory.

## Generate fresh data and rerun full training/search

From R3:

```bash
python rerun_fixed_chain.py --seed 3101
```

Repeat for seeds 3101 through 3110, or run all ten and score:

```bash
python rerun_fixed_chain.py --score
```

Fresh outputs go to `outputs/fixed_chain/`; `--output` changes the destination. A complete run performs 310 product fits and 360 searches and is substantially more expensive than numerical replay. Existing completed outputs are reused. The data generation, selection grids, fitting, optimized complete Fourier evaluator, continuation search and scoring implementations are included. CPU thread count is two. Numerical replay tolerates floating-point differences between NumPy versions, with the stored paired starts checked exactly against the saved traces. Fresh stochastic refits can vary with the numerical environment.
