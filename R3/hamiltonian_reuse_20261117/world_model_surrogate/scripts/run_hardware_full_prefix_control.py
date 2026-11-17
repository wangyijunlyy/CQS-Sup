"""CPU-only validation-matched full-prefix control on six frozen hardware datasets.

No hardware module, Qiskit, Torch, credential store, or network client is used.
The pure feature functions mirror run_ibm_hardware_tfim.py; regression tests
extract only its inspected pure function ASTs without importing that module.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results_hardware/full_prefix_control_20260915"
DATASETS = (
    "ibm_www_n4k10_tk4_20260826",
    "ibm_www_n6_k10_tk4_20260827",
    "ibm_www_n8_k10_tk4_20260827",
    "ibm_www_n10_k10_tk4_20260827",
    "ibm_fez_n4k10_tk4_seed20260827",
    "ibm_fez_n4k10_tk4_seed20260828",
)
ALPHAS = (1e-3, 1e-2, .1, 1., 10., 100.)
MODES = {"full_prefix_val": "full", "full_fixed_alpha1": "full",
         "current_layer_val": "current"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [ROOT / "scripts" / name for name in
             ("run_ibm_hardware_tfim.py", "run_hardware_fair_static.py")]
    for name in DATASETS:
        paths.extend(p for p in (ROOT / "results_hardware" / name).rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def control_rows(X, sample_idx, period_idx, mode):
    """Sample-major rows; k is zero-based. No label argument can enter features."""
    if mode not in ("full", "current"):
        raise ValueError("mode must be full or current")
    rows = []
    K, d = X.shape[1:]
    for s in sample_idx:
        for k in period_idx:
            if not 0 <= k < K:
                raise ValueError("prefix index out of range")
            if mode == "full":
                row = np.zeros(K * d + 1, dtype=np.float64)
                row[:(k + 1) * d] = X[s, :k + 1].reshape(-1)
                row[-1] = (k + 1) / K
            else:
                row = np.concatenate([X[s, k], [(k + 1) / K]])
            rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def pair_indices(d, max_pairs=600):
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(123)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_pairs, replace=False)]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def trig_features(X, max_pairs=600):
    X = np.asarray(X, dtype=np.float64)
    cols = [np.ones((X.shape[0], 1)), np.cos(X), np.sin(X)]
    for i, j in pair_indices(X.shape[1], max_pairs):
        cols.append((np.cos(X[:, i]) * np.cos(X[:, j])).reshape(-1, 1))
        cols.append((np.sin(X[:, i]) * np.sin(X[:, j])).reshape(-1, 1))
    return np.hstack(cols)


def fit_candidates(phi_train, y_train, phi_val, y_val):
    """Choose by early validation MSE; retain train-only fits, first alpha wins ties."""
    records, fitted = [], {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(phi_train, y_train)
        val_mse = float(np.mean((model.predict(phi_val) - y_val) ** 2))
        records.append({"alpha": alpha, "val_mse": val_mse})
        fitted[alpha] = model
    best = min(records, key=lambda item: item["val_mse"])
    return fitted[best["alpha"]], {"selected_alpha": best["alpha"], "validation": records}, fitted[1.]


def fit_controls(X, y, train, val, train_k):
    early = np.arange(train_k)
    # Slice labels before flattening: neither test nor late labels reach any fit.
    y_train = np.asarray(y[train, :train_k], dtype=np.float64).reshape(-1, y.shape[2])
    y_val = np.asarray(y[val, :train_k], dtype=np.float64).reshape(-1, y.shape[2])
    models, details = {}, {}
    for name, mode in (("full_prefix_val", "full"), ("current_layer_val", "current")):
        phi_train = trig_features(control_rows(X, train, early, mode))
        phi_val = trig_features(control_rows(X, val, early, mode))
        models[name], details[name], fixed = fit_candidates(phi_train, y_train, phi_val, y_val)
        if mode == "full":
            models["full_fixed_alpha1"] = fixed
    return models, details


def constant_diagnostic(train_features, late_features, tolerance=1e-12):
    constant = np.ptp(train_features, axis=0) <= tolerance
    changed = np.any(np.abs(late_features - train_features[0]) > tolerance, axis=0)
    active = constant & changed
    return {
        "tolerance": tolerance,
        "feature_count": int(len(constant)),
        "train_constant_count": int(constant.sum()),
        "train_constant_late_active_count": int(active.sum()),
        "train_constant_late_active_fraction": float(active.mean()),
        "train_constant_indices": np.flatnonzero(constant).tolist(),
        "train_constant_late_active_indices": np.flatnonzero(active).tolist(),
    }


def metric_block(y_true, y_pred):
    yt, yp = y_true.reshape(-1), y_pred.reshape(-1)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    return {"mae": float(np.mean(np.abs(yt - yp))), "rmse": rmse,
            "R(h)": rmse * rmse,
            "r2": float(r2_score(yt, yp)) if ss_tot >= 1e-12 else None,
            "pearson": float(np.corrcoef(yt, yp)[0, 1]) if np.std(yt) > 0 and np.std(yp) > 0 else None}


def evaluate(y, pred, test):
    return {
        **{name: metric_block(y[test, window], pred[test, window]) for name, window in
           (("early_test", slice(0, 4)), ("late_test", slice(4, 10)), ("all_test", slice(0, 10)))},
        "period_mae": [float(np.mean(np.abs(y[test, k] - pred[test, k]))) for k in range(10)],
    }


def load_dataset(name):
    path = ROOT / "results_hardware" / name
    with np.load(path / "dataset.npz", allow_pickle=False) as data:
        X, y = data["X"], data["y"]
        splits = {k: data[k + "_idx"] for k in ("train", "val", "test")}
    meta = json.loads((path / "metadata.json").read_text())
    cfg = meta["config"]
    if (cfg["K"], cfg["train_k"]) != (10, 4):
        raise ValueError("expected stored k1..4 / k5..10 protocol")
    if X.shape != (48, 10, 2 * cfg["N"] - 1) or y.shape != X.shape:
        raise ValueError("unexpected dataset shape")
    for key, expected in (("train", np.arange(24)), ("val", np.arange(24, 32)), ("test", np.arange(32, 48))):
        if not np.array_equal(splits[key], expected):
            raise ValueError("stored split differs from approved 24/8/16 protocol")
    for array in (X, y, *splits.values()):
        array.setflags(write=False)
    info = {"id": name, "source": str(path.relative_to(ROOT)), "N": cfg["N"],
            "seed": cfg["seed"], "backend": meta["metadata"]["backend"],
            "dataset_sha256": sha256(path / "dataset.npz"),
            "splits": {key: idx.tolist() for key, idx in splits.items()}}
    return X, y, splits, info


def save_model(path, model, mode, X):
    raw_dim = X.shape[2] * (X.shape[1] if mode == "full" else 1) + 1
    np.savez_compressed(path, coef=model.coef_, intercept=model.intercept_,
                        alpha=model.alpha, mode=mode, K=X.shape[1], d_layer=X.shape[2],
                        pairs=pair_indices(raw_dim), raw_dim=raw_dim)


def evaluate_dataset(out, info):
    name = info["id"]
    X, y, splits, _ = load_dataset(name)
    destination = out / name
    source = ROOT / info["source"]
    predictions = {}
    for method, mode in MODES.items():
        phi = trig_features(control_rows(X, np.arange(len(X)), np.arange(10), mode))
        with np.load(destination / (method + "_model.npz"), allow_pickle=False) as model:
            predictions[method] = (phi @ model["coef"].T + model["intercept"]).reshape(y.shape)
    predictions["saved_gru"] = np.load(source / "pred_latent_gru.npy", allow_pickle=False)
    metrics = {name: evaluate(y, pred, splits["test"]) for name, pred in predictions.items()}
    np.savez_compressed(destination / "predictions.npz", **predictions)
    references = json.loads((source / "metrics.json").read_text())
    fair = json.loads((source / "fair_static_metrics.json").read_text())
    checks = {}
    for method, stored_name in (("full_fixed_alpha1", "static_trig_ridge"),
                                ("current_layer_val", "static_step_trig_ridge"),
                                ("saved_gru", "latent_gru")):
        stored = np.load(source / ("pred_" + stored_name + ".npy"), allow_pickle=False)
        delta = float(np.max(np.abs(predictions[method] - stored)))
        np.testing.assert_allclose(predictions[method], stored, atol=1e-10, rtol=1e-10)
        windows = ("late_test",) if method == "current_layer_val" else ("early_test", "late_test", "all_test")
        score_delta = 0.
        for window in windows:
            target = fair["metrics"][stored_name] if method == "current_layer_val" else references[stored_name][window]
            for key in ("mae", "R(h)", "r2"):
                a, b = metrics[method][window][key], target[key]
                np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)
                score_delta = max(score_delta, abs(a - b))
        checks[method] = {"max_abs_prediction_difference": delta, "max_abs_score_difference": score_delta}
    selected = info["selection"]["current_layer_val"]
    if selected["selected_alpha"] != fair["trig_detail"]["selected_alpha"]:
        raise AssertionError("current-layer alpha does not reproduce saved validation selection")
    np.testing.assert_allclose([r["val_mse"] for r in selected["validation"]],
                               [r["val_mse"] for r in fair["trig_detail"]["validation"]], atol=1e-12)
    diagnostics = {}
    for mode in ("full", "current"):
        early = control_rows(X, splits["train"], np.arange(4), mode)
        late = control_rows(X, splits["test"], np.arange(4, 10), mode)
        diagnostics[mode] = {"raw": constant_diagnostic(early, late),
                             "trig": constant_diagnostic(trig_features(early), trig_features(late)),
                             "period_trig_late_active_count": []}
        train_phi = trig_features(early)
        for k in range(10):
            period_phi = trig_features(control_rows(X, splits["test"], [k], mode))
            diagnostics[mode]["period_trig_late_active_count"].append(
                constant_diagnostic(train_phi, period_phi)["train_constant_late_active_count"])
    info.update(metrics=metrics, reproduction=checks, feature_diagnostics=diagnostics)
    write_json(destination / "metrics.json", info)
    return info


def aggregate(rows):
    result = {}
    for method in rows[0]["metrics"]:
        result[method] = {}
        for window in ("early_test", "late_test", "all_test"):
            result[method][window] = {}
            for metric in ("mae", "R(h)", "r2"):
                values = [r["metrics"][method][window][metric] for r in rows]
                result[method][window][metric] = {"mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None, "n": len(values)}
    return result


def write_report(out, summary):
    lines = ["# Offline Hardware Full-Prefix Control", "",
             "Six unique stored datasets. CPU only; no QPU submission, hardware imports, credentials, or GRU retraining.",
             "All six selections and train-only ridge coefficients were frozen before test evaluation.",
             "Original ordered zero-padded prefix + k/K; unchanged trig map and deterministic 600-pair cap.",
             "Train/validation/test controls: 24/8/16, copied from each stored dataset; train and validation k=1..4 only.",
             "Validation objective: MSE; candidates 0.001, 0.01, 0.1, 1, 10, 100; first minimum wins; no train+validation refit.",
             "R(h) is flattened MSE, not correlation; R2 is flattened across test controls, periods and observables.", "",
             "## Per-Dataset Errors", "",
             "| N | Seed | Backend | Method | Alpha | Early MAE | Late MAE | Late R(h) | Late R2 |",
             "|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    csv_rows = []
    for row in summary["datasets"]:
        for method, metrics in row["metrics"].items():
            alpha = row["selection"].get(method, {}).get("selected_alpha", 1. if method == "full_fixed_alpha1" else "saved")
            m = metrics["late_test"]
            lines.append(f"| {row['N']} | {row['seed']} | {row['backend']} | {method} | {alpha} | "
                         f"{metrics['early_test']['mae']:.9f} | {m['mae']:.9f} | {m['R(h)']:.9f} | {m['r2']:.9f} |")
            for window in ("early_test", "late_test", "all_test"):
                csv_rows.append({"id": row["id"], "N": row["N"], "seed": row["seed"], "backend": row["backend"],
                                 "method": method, "alpha": alpha, "window": window, **metrics[window]})
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines += ["", "## Validation Candidate Curves", "",
              "| N | Seed | Method | 0.001 | 0.01 | 0.1 | 1 | 10 | 100 | Selected |",
              "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    curves = []
    for row in summary["datasets"]:
        for method, selection in row["selection"].items():
            values = " | ".join(f"{r['val_mse']:.10f}" for r in selection["validation"])
            lines.append(f"| {row['N']} | {row['seed']} | {method} | {values} | {selection['selected_alpha']} |")
            curves.extend({"id": row["id"], "method": method, **r} for r in selection["validation"])
    write_json(out / "candidate_curves.json", curves)
    lines += ["", "## Train-Constant / Late-Active Features", "",
              "A late-active feature was constant on train k=1..4 but differs from that constant on test k=5..10 (tolerance 1e-12).",
              "Cosines of padded controls are constant one, not zero. Counts include constant trigonometric interactions.", "",
              "| N | Seed | Raw active / total | Trig active / total | Trig train-constant | Step trig active |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in summary["datasets"]:
        d = row["feature_diagnostics"]
        a, b = d["full"]["raw"], d["full"]["trig"]
        lines.append(f"| {row['N']} | {row['seed']} | {a['train_constant_late_active_count']}/{a['feature_count']} | "
                     f"{b['train_constant_late_active_count']}/{b['feature_count']} | {b['train_constant_count']} | "
                     f"{d['current']['trig']['train_constant_late_active_count']} |")
    lines += ["", "## Interpretation and Limitations", "",
              "This aligns available ordered prefix information and held-out validation, not invariance to depth. "
              "Position-specific late control coordinates and their interactions remain unidentifiable from early-only training; "
              "k/K also extrapolates beyond its training range. Alpha tuning does not repair either limitation.",
              "The unchanged trig basis has no arbitrary pooling and retains its capped pair subsampling; this does not establish "
              "equivalent function classes, capacity, inductive bias, or tuning budgets between static ridge and the saved GRU.",
              "Metrics are against stored noisy hardware estimates, not noiseless truth. No fresh hardware replication was performed.",
              "N=4 has three seeds from separate acquisition runs on ibm_fez; N=6/8/10 each has one seed. "
              "Seed and acquisition-run differences are not separated. "
              "The six-dataset macro average is descriptive across heterogeneous sizes, not an independent-seed significance test. "
              "The N=4 scaling dataset occurs only once in the six-dataset aggregate. "
              "All six full-prefix selections reach the upper endpoint alpha=100 of the prespecified grid; "
              "no expanded grid was explored after observing test results.", "",
              "## Artifacts and Replay", "",
              "`selection_frozen.json`: protocol, original SHA256 hashes, all six validation choices, model SHA256 hashes and runtime versions.",
              "`summary.json`: full precision metrics, per-period MAE, diagnostics with feature indices, reproduction deltas and grouped aggregates.",
              "`metrics.csv`: every dataset/method/window; `candidate_curves.json`: early-validation MSE only, no candidate test curves.",
              "Each dataset directory: `predictions.npz` (all 48 controls x 10 prefixes x observables), three `*_model.npz` files, and `metrics.json`.",
              "Replay: construct `control_rows(X, arange(48), arange(10), mode)`; apply `trig_features`; "
              "compute `(phi @ coef.T + intercept).reshape(y.shape)`. NPZ includes ordered pair indices, alpha, K and layer width. "
              "Feature order is [1, cos(raw), sin(raw), cos_i*cos_j, sin_i*sin_j for each saved pair]. No scaling or clipping.",
              "The GRU is represented by unchanged saved predictions, not new coefficients or a newly trained checkpoint.",
              "Run with `.venv_ibm/bin/python -B world_model_surrogate/scripts/run_hardware_full_prefix_control.py`; "
              "a subsequent replay run may use `--out` with a fresh subdirectory under this result directory.",
              "Verify with `.venv_ibm/bin/python -B world_model_surrogate/scripts/test_hardware_full_prefix_control.py`.", ""]
    (out / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(DEFAULT_OUT.resolve()):
        parser.error("outputs must stay under the owned full_prefix_control_20260915 directory")
    if out.exists() and any(out.iterdir()):
        parser.error("output must be empty; preserve frozen results and use a fresh subdirectory")
    out.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    protocol = {"offline": True, "device": "cpu", "blas_threads": 1,
                "train_periods": [1, 2, 3, 4], "validation_periods": [1, 2, 3, 4],
                "late_test_periods": [5, 6, 7, 8, 9, 10], "alpha_candidates": list(ALPHAS),
                "selection": "minimum held-out early-validation MSE; first minimum wins",
                "refit_on_validation": False, "test_used_for_selection": False,
                "observations_in_features": False, "future_controls_visible": False,
                "input": "ordered FULL zero-padded control prefix plus k/K; original trig_features",
                "depth_invariance_claimed": False}
    infos = []
    with threadpool_limits(limits=1):
        for name in DATASETS:
            X, y, splits, info = load_dataset(name)
            models, info["selection"] = fit_controls(X, y, splits["train"], splits["val"], 4)
            destination = out / name
            destination.mkdir()
            for method, model in models.items():
                save_model(destination / (method + "_model.npz"), model, MODES[method], X)
            infos.append(info)
            print(json.dumps({"phase": "validation_only", "dataset": name,
                              "alphas": {k: v["selected_alpha"] for k, v in info["selection"].items()}}), flush=True)
        if len({r["dataset_sha256"] for r in infos}) != 6:
            raise ValueError("expected exactly six unique datasets")
        frozen = {"protocol": protocol, "source_sha256": before, "datasets": infos,
                  "model_sha256": {str(p.relative_to(out)): sha256(p) for p in sorted(out.glob("*/*_model.npz"))},
                  "runner_sha256": sha256(Path(__file__)),
                  "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                  "versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "sklearn": sklearn.__version__, "platform": platform.platform()}}
        write_json(out / "selection_frozen.json", frozen)
        # Only after ALL selections and coefficients are persisted, open saved
        # predictions/scores and evaluate test labels. No fitting occurs below.
        rows = [evaluate_dataset(out, info) for info in infos]
    after = source_hashes()
    if before != after:
        raise RuntimeError("historical inputs changed during the run")
    summary = {"completed": True, "protocol": protocol, "datasets": rows,
               "original_inputs_unchanged": True, "source_file_count": len(before),
               "selection_frozen_sha256": sha256(out / "selection_frozen.json"),
               "all_six_macro": aggregate(rows),
               "by_N": {str(n): aggregate([r for r in rows if r["N"] == n]) for n in (4, 6, 8, 10)}}
    if any(n.startswith(("qiskit", "torch", "run_ibm_hardware", "run_hardware_fair_static")) for n in sys.modules):
        raise RuntimeError("unexpected hardware/training module imported")
    write_json(out / "summary.json", summary)
    write_report(out, summary)
    print(json.dumps({"completed": True, "out": str(out), "original_inputs_unchanged": True,
                      "all_six_macro_late": {k: v["late_test"] for k, v in summary["all_six_macro"].items()}}, indent=2))


if __name__ == "__main__":
    main()
