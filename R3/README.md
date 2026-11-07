# R3: additional experiments and reproducibility materials

This folder provides the experiment code, the actual datasets and model checkpoints, and the saved predictions and measurement/search traces used in the R3 response. The main artifact and an additional TFIM training-input bundle are attached to the `r3-repro-20261108` release of this same repository. `fetch_artifacts.py` downloads and extracts them automatically; no author workstation or server is required.

## Reproduce the results

Use Python 3.12. From this `R3` folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_artifacts.py
python reproduce.py --suite all
```

The downloader derives the repository name from the checkout's Git origin. If the folder was downloaded without Git metadata, pass `--repository OWNER/REPOSITORY` to the downloader. Both the code and the release asset must be retained when making a mirror.

The artifact contains 8,492 files, including **456 checkpoint files**. It includes acquired/simulated labels, exact evaluation targets, split indices, Fourier/NC coefficients, neural-network weights, validation histories, and saved optimization and measurement records. `artifact_inventory.json` enumerates them. The download is checked against the published archive checksum before extraction. No credentials or hardware-service account is needed to replay the hardware measurements.

The default reproduction command recomputes numerical evidence from these inputs. It evaluates saved neural checkpoints, reconstructs energy scores and acceptance decisions, and checks the measurement budgets. It does not rerun the expensive training or acquire new QPU measurements. A successful run writes `reproduction_report.json`.

Individual suites are available:

```bash
python reproduce.py --suite hardware
python reproduce.py --suite w1
python reproduce.py --suite w2
python reproduce.py --suite query
python reproduce.py --suite chain
```

## Experiments and entry points

All paths below are relative to this folder. The directory layout is retained so that the original experiment programs resolve their data automatically.

| Evidence | Programs in `paper_iclr2027/rebuttal_20261107/` |
| --- | --- |
| Constant predictors, hardware regularization, endpoint diagnosis, exact-objective controls | `run_evidence.py` |
| Validation-tuned NC for VQE and FSPT | `tune_nc.py`, `run_W1_fspt_matrix.py` |
| Shared Fourier-coefficient regularization | `run_W1_common_penalty.py`, `refine_W1_common_penalty.py` |
| Exact-label precision control | `precision_control.py` |
| Mask, energy-error and TFIM analysis | `analyze_W1_existing.py` |
| Training first on 12 periods and then all 24 | `run_W2_curriculum.py`, `analyze_W2.py` |
| Stronger-coupling grids and tuned learned baselines | `confirm_warmstart.py`, `stronger_grid_controls.py`, `local_refinement.py`, `local_strong_controls.py` |
| Product-rank selection | `chain_rank_tuning.py` |
| Identity-layer initialization, staged search and local measurement correction | `chain_depth_recovery.py`, `chain_local_calibration.py` |
| Saved-search ranking reversals and same-candidate correction errors | `analyze_chain_depth_recovery.py`, `analyze_chain_mechanism.py` |
| Analytic representability and same-history execution | `mechanism_checks.py` |
| Dense implementation costs | `cost_audit.py` |
| Independent fits under an actual training-time budget | `resource_query.py`, `analyze_resource_query.py` |
| Finite-shot SPSA/parameter-shift Adam and total-budget curves | `resource_chain.py`, `analyze_resource_chain.py` |

The TFIM supplement also includes the original measured training labels and checkpoints needed to refit the tuned NC baseline. Run `python retrain_tfim.py --check-inputs` to validate all twelve input sets, or `python retrain_tfim.py --n 8 --seed 42` to perform a fresh fit in `outputs/tfim/`. The bundled reference NC implementation retains its third-party license.

The corresponding model, circuit, fitting and data-generation implementations are included under `world_model_surrogate/src/` and `world_model_surrogate/scripts/`. `code_manifest.json` lists the included files.

## Run training or search again

The released checkpoint and trace directories preserve the reported runs. Programs that detect completed outputs reuse them. To create fresh runs, keep a copy of the released outputs and move the corresponding output directory aside before invoking a program with a fixed destination.

For the independent equal-time comparison, the input dataset is included and the destination can be specified directly:

```bash
python paper_iclr2027/rebuttal_20261107/resource_query.py \
  --seed 1001 \
  --data world_model_surrogate/results_supplementary/e2/confirmation/seed1001/dataset.npz \
  --out outputs/query/seed1001 --seconds 240 --device cuda
```

Repeat for seeds 1001 through 1005. Each seed runs all three full models. The published timing experiment uses a dedicated RTX 4090, sequential arms with counterbalanced order, synchronized training/validation/checkpoint selection, and 240 seconds per arm. Retraining under the same wall time on different hardware can produce different optimizer-update counts and accuracy. The released checkpoints support numerical replay independently of that hardware timing.

Other full experiment commands, from this folder:

```bash
python paper_iclr2027/rebuttal_20261107/run_W1_common_penalty.py
python paper_iclr2027/rebuttal_20261107/refine_W1_common_penalty.py
python paper_iclr2027/rebuttal_20261107/run_W1_fspt_matrix.py
python paper_iclr2027/rebuttal_20261107/run_W2_curriculum.py
python paper_iclr2027/rebuttal_20261107/confirm_warmstart.py
python paper_iclr2027/rebuttal_20261107/chain_rank_tuning.py
python paper_iclr2027/rebuttal_20261107/chain_depth_recovery.py
python paper_iclr2027/rebuttal_20261107/chain_local_calibration.py
python paper_iclr2027/rebuttal_20261107/resource_chain.py --seed 45
python paper_iclr2027/rebuttal_20261107/resource_chain.py --seed 46
python paper_iclr2027/rebuttal_20261107/resource_chain.py --seed 47
```

The two-stage FSPT experiment starts from the included 12-period checkpoint and retains its pretraining cost in the reported protocol. Chain local correction starts from the explicit energy -3 circuit and retains the original global models. Its additional measurements are separate from the fixed-data search. Exact energies are used for scoring saved outputs, except in the explicitly named exact-objective diagnostic.

## Checks and numerical interpretation

```bash
python -m unittest discover \
  -s paper_iclr2027/rebuttal_20261107 -p test_resource_controls.py -v
```

This checks matched-history predictions and gradients, independent circuit probabilities, parameter-shift queries, and shot accounting. Reported uncertainty is across independent dataset/run means; optimization restarts are averaged within each run. Floating-point roundoff can vary across supported platforms. Hardware timing should be compared on the stated hardware and execution settings.

Host locations and identifying account metadata have been removed from the distributed records. Numerical data arrays, model weights, seed values, train/validation/test roles and experimental settings are preserved. Paths in the programs and documentation refer to files within this package.
