Sequential Hamiltonian reuse experiment

This package evaluates unchanged CQS-P models on10 different TFIM Hamiltonians,
compared with SPSA that may reuse all its own historical measurements and controls.
No QPU measurements were acquired. All shot counts are finite-shot simulations.

Tasks: grid9q/L2/shared3-active-control and chain6q/L2/16-active-control circuits.
Ten existing independent dataset/model fits per task are used without retraining.
h=-0.5, J=-0.05,-0.10,...,-0.50. The Hamiltonians have different coefficient
ratios, with no mere scalar multiples or J-sign-symmetry duplicates. Two orders:
smooth, and jumping between small/large coupling magnitudes.

CQS: rank32/Q4 grid and rank32/Q1 chain; the original complete fitted weights
are used.100 classical Adam updates, lr0.1 per target. No online quantum queries.
The once-only acquisition is4M shots/grid,24k shots/chain, charged at every prefix.

SPSA-low uses a40k-shot grid pilot or a free known analytic chain reference.
SPSA-shared receives the entire same initial measured pool as CQS, paying the
same initial cost. Each order, pool regime and budget owns an independent history.
At every Hamiltonian, the best historical mean+2SE control includes all previous
retained endpoints, with19 perturbed starts plus the unperturbed center. This
avoids forcing all starts onto the analytic chain saddle at later tasks.
Exact-control measurements are cached and topped up only when needed. Xsum and
ZZsum count histograms are sufficient statistics for this Hamiltonian family.
They retain both the energy estimate and its within-group variance under new J.

SPSA gain/c retain the earlier frozen values: grid0.05/0.05, chain0.2/0.2.
Shots/group100 or500 are selected using old development datasets45/46 and a
separate1000-shots/group holdout readout after all200 sequence endpoints are
selected. These holdout labels never feed search, retention or warmstarts.
All choices are frozen before formal workloads. Development costs are separate.
Each of4 online per-target caps0/160k/640k/1280k is rerun independently for all
10 targets. A cheaper arm never inherits a more expensive arm's observations.
The zero-cap arm is measured-history selection without new queries.

All20 endpoints are averaged inside a dataset. Independent dataset/model fits
are the statistical units, n=10 per task. Neither20 endpoints nor10 Hamiltonians
are counted as additional independent replicates. Intervals are descriptive t95.

Shots count actual state preparations. Measured-control counts include precision
topups and are not basis-resolved circuit counts. Feedback rounds count serial
measurement-dependent batches: both SPSA probes,20 trajectories and both grouped
bases can share one round; cached reads take0 rounds. Fixed initial acquisition
counts1 round. These are logical rounds, not real hardware jobs or latency.

Reported exact energies are post-search simulation assessments, after all searches
and chronological audits. Final hardware certification measurements are excluded
symmetrically. Zero online CQS calls alone do not imply quality-matched shot savings.
The cost frontier uses post-hoc average-energy tolerance0.020 on the tested budget
grid. It is descriptive, not an oracle stopping rule or the optimal SPSA frontier.

Install the root requirements.txt in a Python3.10 virtual environment. From the
extracted root, replay all recorded histories, resource counts and exact energies:

  python paper_iclr2027/rebuttal_20261107/Hamiltonian_reuse_20261117/reproduce.py

Regenerate every workload from the released unchanged models, original measured
datasets and frozen SPSA settings in a separate empty folder:

  python paper_iclr2027/rebuttal_20261107/Hamiltonian_reuse_20261117/reproduce.py --fresh --out fresh_reuse --workers 2

This does not retrain the existing models. It reruns all360 sequential workloads.
The source and measured data used to fit these models were released with the
earlier JJuz_followup_20261117 artifact. This package includes their original
datasets, full checkpoints and fitting source for inspection, and the additional
development readouts and complete reuse histories needed for this experiment.

protocol.json, frozen_selection.json: design, configuration and model/data hashes.
audit.json: chronological replay of measurement histories, updates and retention.
verification.json: full scoring, independent Qiskit spot checks and limitations.
per_target.csv: every target/dataset/method result.
prefix_by_dataset.csv: cumulative costs and quality at1/2/5/10 targets.
summary.json, paired_contrasts.json: dataset-level summaries and paired contrasts.
descriptive_frontier.json: sampled quality-cost frontier, including zero-budget arms.
development/: all tuning readouts; runs/: complete final workload evidence.
