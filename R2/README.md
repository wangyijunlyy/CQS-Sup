# R2: coupled Fourier-product and recurrent surrogate experiments

This package implements the architecture-level combination of CQS-P and CQS-R. Observable-dependent masked Fourier products enter a shared GRU at every circuit period. A running product state and recurrent state jointly predict the observables. The FSPT adapter reuses those states across requested prefixes. The TFIM adapter preserves the original scalar supervision, and the VQE adapter uses the original measured moments.

The package contains the original experiment code, frozen configurations, validation histories, per-run results, and numerical inputs for both the full main-benchmark comparison and the earlier matched-time feasibility experiment. Data, model checkpoints, saved predictions, and search trajectories are supplied through a Release asset of this repository. No private machine, credential or QPU access is needed.

## Reproduce the reported results on CPU

Use Python 3.12. From `R2`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_artifacts.py
python reproduce.py --suite all
```

For a source ZIP without Git metadata, provide `--repository OWNER/REPOSITORY` to the downloader. `--archive PATH` uses an already downloaded asset. `python fetch_artifacts.py --check` checks required file presence.

The replay recomputes all 66 main-model test metrics from the complete saved test arrays, evaluates each of those checkpoints on 16 test inputs, independently evaluates all 660 terminal circuit energies from 33 searches, and audits the original baseline predictions and endpoints. It also recomputes the 20 feasibility fits' saved metrics, evaluates their checkpoints on 16 test inputs each, and verifies the displayed means and sample standard deviations. CPU checkpoint replay uses an explicit cross-backend tolerance of 0.000020 plus relative tolerance 0.000020. The report records the actual maximum errors at full precision. Replay results are written to `reproduction_report.json`.

This is numerical replay of completed experiments. Full training and search commands are given below. The original GPU validation replay audits remain in each experiment's `verification.json`.

## Experiment and evidence map

All paths below are relative to `R2`.

| Evidence | Code and records | Original protocol |
| --- | --- | --- |
| Main TFIM comparison, Table 65 | `paper_iclr2027/rebuttal_20261107/hybrid_main_20261108/` | 4 system sizes × 3 original splits, 12 fits |
| Main FSPT comparison, Table 66 | Same directory | 4 sizes × 3 horizons × 3 original splits, 36 fits |
| Main VQE comparison, Tables 67–68 | Same directory | 6 circuit settings × 3 original datasets, 18 fits and 33 searches, retaining 20 outputs per search |
| Matched-time feasibility, Tables 63–64 | `paper_iclr2027/rebuttal_20261107/hybrid_feasibility_20261108/` | 5 existing independent pools × 4 architectures, 120 seconds per fit |

Table numbers refer to the English manuscript. `main_models.py` implements the three main-task adapters. `models.py` in the feasibility directory implements the joint architecture, parameter-matched recurrent reference, product reference, and additive control. `analyze_main.py` and `analyze_hybrid.py` implement the original audits and scoring stages. The required circuit simulators, masks and fitting routine are included under `world_model_surrogate/`.

In each experiment directory, `protocol.json` records the settings, `rows.json` retains the individual runs, `summary.json` retains aggregates, and `实验结果.md` presents results to three decimal places as mean ± sample SD. Raw numerical records retain their original precision. Training histories and selected checkpoints are under `fits/` for the main experiment and `pools/` for feasibility. Main search traces are under `search/`. The original baseline inputs are under `world_model_surrogate/results_*`.

The main model has Fourier rank 32 and recurrent width 64. Main training uses Adam with learning rate 0.001, weight decay 0.0001 and batch size 32. The frozen epoch and patience limits are recorded for every cell. Checkpoints are selected with validation data, with all 66 fits frozen before new-model test scoring. VQE uses the original 20 starts and 100 direct Adam steps, retaining all outputs before exact energy scoring. Main training is epoch-based; the separate feasibility experiment matches wall time.

The main joint model has lower mean MAE in 3/4 TFIM settings, improves on CQS-P in 12/12 FSPT settings and on original CQS-R in 6/12. Its direct VQE search does not outperform CQS-P+C in the six primary settings. These task-dependent conclusions are retained in the supplied results. TFIM/FSPT runs are paired splits of the original pools; they are not new independent datasets. VQE retains the original independent datasets. The 24-period staged CQS-R references use additional 12-period pretraining and are identified separately.

## Structural and audit tests

Run these two commands as separate processes to avoid collisions between experiment-local module names:

```bash
python -m unittest discover -s paper_iclr2027/rebuttal_20261107/hybrid_main_20261108 -p 'test*.py' -v
python -m unittest discover -s paper_iclr2027/rebuttal_20261107/hybrid_feasibility_20261108 -p 'test*.py' -v
```

The 13 tests cover coupled gradient paths, masked-factor behavior, prefix output/gradient equivalence, causal ordering, task adapters, parameter matching, original split membership, complete search traces, and owned-process cleanup.

## Full retraining and search on an idle NVIDIA GPU

The original fits used Python 3.10, PyTorch 2.6.0 with CUDA 12.4, NumPy 2.0.2, two CPU threads and an RTX 4090. `requirements-training-cuda.txt` specifies that training stack. A fixed 120-second fit is hardware-dependent. Main comparisons preserve the original datasets, seeds, training caps and supervision; they do not impose equal wall time across all original baselines. The supplied baseline records are replayed rather than retrained by the commands below.

First download the artifacts. Then make a new run directory, which preserves the reported-run package and copies the original source inputs without the completed fusion outputs:

```bash
python prepare_fresh.py ../R2-fresh
cd ../R2-fresh
```

In the CUDA environment, run the 66-cell main matrix sequentially on a GPU that is free of other compute jobs:

```bash
MAIN=paper_iclr2027/rebuttal_20261107/hybrid_main_20261108
python "$MAIN/analyze_main.py" baselines
python "$MAIN/run_main.py" --worker 0 --workers 1
python "$MAIN/run_main.py" --freeze
python "$MAIN/analyze_main.py" audit
python "$MAIN/analyze_main.py" search
python "$MAIN/analyze_main.py" score
```

For the original feasibility experiment:

```bash
FEAS=paper_iclr2027/rebuttal_20261107/hybrid_feasibility_20261108
for seed in 1001 1002 1003 1004 1005; do
  python "$FEAS/run_hybrid.py" --seed "$seed"
done
python "$FEAS/run_hybrid.py" --freeze
python "$FEAS/analyze_hybrid.py"
```

Training refuses to overwrite an interrupted fit. The provided published records include the completed training selection and scoring stages. No full retraining was repeated merely to package these files. The CPU replay checks distributed results and checkpoints directly.

`artifact_inventory.json` lists the numerical asset contents. Machine-local paths in JSON metadata have been made relative; model, dataset, prediction and search arrays are unchanged. Remote-job launchers and workstation connection settings are excluded.
