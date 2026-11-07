# R1: experiment code, data and numerical reproduction

This folder contains the experiments supporting the R1 response to **Compositional Quantum Surrogates: Circuit-Guided Representations for Quantum Surrogate Learning**. It includes the actual model checkpoints, acquired/simulated labels, validation histories, Fourier coefficients, exact evaluation targets, and complete saved search/measurement records. Large numerical files are distributed as a Release asset of this repository. Source code and frozen configurations are in Git.

## Reproduce the reported numerical results

Use **Python 3.12**. From this `R1` directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_artifacts.py
python reproduce.py --suite all
```

No private server, author workstation, account credential or QPU access is required. The downloader derives the repository from the checkout's Git origin. For a source ZIP without Git metadata, use `python fetch_artifacts.py --repository OWNER/REPOSITORY`. Retain both the source and Release asset when mirroring the package.

The default command is **numerical replay of the reported runs**. It recomputes errors from raw predictions/targets, re-evaluates saved neural checkpoints, reconstructs Fourier fits and validation choices, evaluates terminal circuit energies, and checks measured acceptance and shot budgets. It writes `reproduction_report.json` and additional results under `outputs/`. It does not repeat every expensive training run. All replay suites run on CPU. The fixed-data chain audit includes all 310 saved product fits and 360 searches across the ten new datasets.

The parallel suite recomputes all 45 published test metrics from the saved predictions. It also evaluates all 45 checkpoints on CPU, with strict prediction comparisons for the 25 GRU/MLP checkpoints. The 20 Transformer CPU outputs are reported as cross-backend diagnostics: the original PyTorch 2.6 CUDA fused Transformer path differs from its CPU/unfused path. A same-checkpoint diagnostic reproduced the saved CUDA predictions exactly, while disabling that CUDA fast path reproduced the observed CPU discrepancy. The model and the tolerance are not changed to hide this difference. To strictly replay all 45 checkpoints, use a Python 3.10 environment with `requirements-parallel-cuda.txt` on an NVIDIA GPU:

```bash
python reproduce.py --suite parallel --device cuda
```

`artifact_inventory.json` enumerates the distributed numerical files. `artifacts.json` gives the asset name, size and archive digest. `code_manifest.json` lists the source. Some original analysis programs regenerate their numerical summary files in place; raw checkpoints, labels and search traces remain the inputs.

## Response-to-experiment map

| Response item | Evidence | Replay suite |
| --- | --- | --- |
| Weakness 1 and mask attribution in Weakness 2 | Table 5: same-mask/full-mask product and complete Fourier | `masks` |
| Same concern | Table 41: common Fourier-coefficient penalty | `common` |
| Question 1 and inference cost in Weakness 2 | Tables 54–55: 30 full joint-MLP/Transformer fits, validation/latency selection and all 45 checkpoint predictions | `parallel` |
| Same concern | Table 52: independent CQS-R/Query-GRU fits with identical histories and 240-second training budgets | `query` |
| Question 2 and chain baselines in Weakness 2 | Ten new fixed-data chain datasets, seeds 3101–3110, both uniform/clipped and reference/wrapped continuation searches | `fixed-chain` |
| Question 2: ansatz quality | Ground lower bound and saved best-found feasible ansatz controls | `reference` |
| Matched grid continuation | Table 32: complete Fourier+C and CQS-P+C on all four grid settings | `grid-continuation` |
| Question 3: improvement beyond a feasible reference | Tables 57–58: 16-control chain, local correction and measured direct optimizers | `local-chain` |
| Same concern | Table 60: five new local-refinement datasets, seeds 2101–2105 | `local-confirmation` |
| Grid energy quality | Table 46: five new stronger-coupling grid datasets, seeds 5101–5105 | `strong-grid` |
| Evaluation cost and scalability scope | Table 59: saved forward/gradient timing repetitions | `timing` |

For example:

```bash
python reproduce.py --suite fixed-chain
python reproduce.py --suite parallel
python reproduce.py --suite local-chain
```

All source and output paths below are relative to this directory. The original relative layout is retained so the scientific programs resolve their inputs without machine-specific path edits.

## Repeat full training and search

`rerun.py` redirects the original programs to a separate output directory. It preserves their full model families, grids, splits, seed rules and optimization settings. Use a new directory for a new run. Existing complete outputs in that directory resume rather than being silently overwritten.

```bash
# Table 5: all five independent datasets and full product/Fourier fits.
python rerun.py masks --out outputs/new_masks --device cpu

# Table 41: original coarse selection followed by the symmetric refined grid.
python rerun.py common --out outputs/new_common

# All 30 parallel fits and a new same-device latency/validation selection.
python rerun.py parallel --out outputs/new_parallel --device cuda

# All 15 full fits under 240 seconds each. Change the device only with the
# understanding that wall-time-limited training depends on that device.
python rerun.py query --out outputs/new_query --device cuda

# All 10 new fixed-data chain pools,31 product fits/pool, both Fourier selectors,
# both search settings, six couplings and 20 outputs/search.
python rerun.py fixed-chain --out outputs/new_fixed_chain

# The exact 32-start BFGS diagnostic, with its original seed and stopping rules.
python rerun_reference.py

# Complete grid Fourier order/ridge selection and matched continuation.
python rerun.py grid-continuation --out outputs/new_grid_continuation

# All 54 product/NC/local-only measurement-augmented chain searches.
python rerun.py local-chain --out outputs/new_local_chain

# All 180 local-confirmation searches, including the original, calibrated,
# and time-limited settings. Actual timing depends on the current CPU.
python rerun.py local-confirmation --out outputs/new_local_confirmation

# Five fresh grid datasets, all three couplings and all reported comparators.
python rerun.py strong-grid --out outputs/new_strong_grid
```

The parallel run reuses the released historical trajectory checkpoints as the comparator for the new parallel fits. `resource_query.py` and `world_model_surrogate/scripts/run_supplement_e2.py` also provide their complete training recipes. The mask run generates new data with the original seeds. The fixed-chain and local-confirmation runs also regenerate their measured records. Other retraining modes use the explicitly released input pools and frozen prior checkpoints where specified in the experiment.

Supported seed subsets are available through `--seeds`, for example:

```bash
python rerun.py fixed-chain --out outputs/new_fixed_chain --seeds 3101
```

The fixed-chain aggregate audit runs only after **all ten pools** and their complete searches exist. A one-seed run is a full run for that pool, not a reproduction of the ten-pool aggregate.

### Full Fourier local correction and exploratory direct optimizers

The original entry points are included under `paper_iclr2027/rebuttal_20261107/reviewer_weak_reject_2/extensions_20261108/`:

```bash
python paper_iclr2027/rebuttal_20261107/reviewer_weak_reject_2/extensions_20261108/run_fourier_chain.py --help
python paper_iclr2027/rebuttal_20261107/reviewer_weak_reject_2/extensions_20261108/direct_jitter.py --help
```

`run_fourier_chain.py --seed 45 --out outputs/new_fourier_chain` fits the complete masked Fourier model and repeats its fixed-data/local searches in a new destination. `direct_jitter.py --seed 45` repeats both measured direct optimizers. Repeat for 45, 46, 47. The direct-optimizer driver has a fixed output location and preserves completed runs; to repeat it after downloading the artifacts, move the corresponding `extensions_20261108/direct_jitter/seed45_*` outputs aside first. The separate `rerun.py local-chain` command concerns the product/NC/local-only branch, and is not a substitute for the Fourier branch.

## Numerical environment and timing

`requirements.txt` pins the tested Python 3.12 replay environment. `requirements-chain-training.txt` records the Python 3.10 environment used for the new chain training/search. Per-run metadata also retain recorded library versions. For the original chain training environment, create a separate Python 3.10 environment and install that file. The broader replay and statistical audits use the Python 3.12 environment.

The published trajectory timings used the same RTX 4090, host-to-host transfers,10 warmups and 40 repetitions. A fresh `parallel` run measures its current device rather than copying those times. The CPU representation benchmark source is `extensions_20261108/benchmark_chain.py`. It measures model evaluation/gradients and excludes training and acquisition. Its results depend on the current processor and numerical libraries. The `timing` replay suite checks the consistency of the published raw timing records.

All reported aggregations use independent dataset/run means. Restarts are averaged within a run. Fixed-data chain searches use 24,000 acquisition shots/pool and zero new search shots. The measurement-augmented chain uses 1,686,000 total shots for pretrained methods. Stronger-coupling grid refinement uses measured warm starts and unsmoothed updates. These protocols are distinct and remain separate in both the code and replay output.

The chain has six qubits, two circuit layers and 16 effective independent control angles. Its one-dimensional designation describes connectivity. The shared-angle two-layer grids have three effective controls. The retained energy conventions are `-J sum(ZZ) - h sum(X)` for chains and `J sum(ZZ) + h sum(X)` for grids, with `h=-0.5`.

Checkpoint replay uses explicit floating-point tolerances. Complete retraining can differ slightly across PyTorch versions, devices and kernels. The saved data, checkpoints and full traces allow the reported numerical evidence to be checked without depending on an exact repetition of stochastic training or machine timing.

## Numerical implementation checks

```bash
python -m unittest discover -s paper_iclr2027/rebuttal_20261107/reviewer_weak_reject_2/fixed_data_chain_20261108 -p 'test_*.py' -v
python -m unittest discover -s paper_iclr2027/rebuttal_20261107/reviewer_weak_reject_2/extensions_20261108 -p test_extensions.py -v
```

These check coefficient penalties, Fourier smoothing, continuation, periodic controls, artifact completeness and deadline eligibility. They complement the full numerical replay.

The distributed content removes personal names, workstation paths, private host/account details and credentials. Numeric arrays, weights, seeds, data roles and measured outcomes are preserved. The hosting account and repository activity are separate from content anonymization.
