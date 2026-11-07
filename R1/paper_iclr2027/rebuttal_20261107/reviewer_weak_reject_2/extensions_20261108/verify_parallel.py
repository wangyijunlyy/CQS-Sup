"""Independently replay persisted checkpoints, selections and all test metrics."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import sys
import json
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]/'world_model_surrogate'
sys.path.insert(0, str(HERE/'parallel'))
from benchmark_parallel import load
from parallel_models import CONFIGS, select_under_budget
from run_parallel import write


def main():
    torch.set_num_threads(2)
    root = HERE/'parallel'
    evaluation = root/'evaluation'
    timings = json.loads((evaluation/'timings.json').read_text())
    frozen = json.loads((evaluation/'selection_frozen.json').read_text())
    rows = json.loads((evaluation/'test_rows.json').read_text())
    summary = json.loads((evaluation/'summary.json').read_text())
    gpu = json.loads((evaluation/'gpu_replay.json').read_text())
    assert gpu['passed'] and gpu['checkpoint_replays'] == 45
    names = ['cqs_r', 'query_gru', 'period_mlp']+list(CONFIGS)
    assert len(rows) == len(timings) == 45 and summary['full_fits'] == 30
    max_prediction_error, max_metric_error = 0., 0.
    for seed in range(1001, 1006):
        with np.load(ROOT/f'results_supplementary/e2/confirmation/seed{seed}/dataset.npz') as a:
            x = a['X'][2400:]
            truth = a['y'][2400:]
        for name in names:
            if name in CONFIGS:
                fit = json.loads((root/'output'/f'seed{seed}'/name/'fit.json').read_text())
                history = json.loads((root/'output'/f'seed{seed}'/name/'history.json').read_text())
                best = min(history, key=lambda r: r['val_mse'])
                assert fit['selected_epoch'] == best['epoch']
                assert fit['best_val_mse'] == best['val_mse']
                assert fit['epochs_run'] <= 300
                if fit['epochs_run'] < 300:
                    assert fit['epochs_run']-fit['selected_epoch'] == 30
            model = load(name, seed, root, 'cpu')
            with np.load(evaluation/f'seed{seed}_{name}.npz') as a:
                np.testing.assert_array_equal(a['exact'], truth)
                prediction = a['prediction']
            with torch.inference_mode():
                replay = model(torch.tensor(x[:16], dtype=torch.float32)).numpy()
            error = float(np.max(abs(replay-prediction[:16])))
            max_prediction_error = max(max_prediction_error, error)
            # CPU and CUDA attention can differ numerically. The authoritative
            # checkpoint replay uses all 400 inputs in the original CUDA setup.
            assert np.isfinite(replay).all(), (seed, name)
            row = next(r for r in rows if r['seed'] == seed and r['name'] == name)
            for key, value in [('test_mae', np.mean(abs(prediction-truth))),
                               ('test_mse', np.mean((prediction-truth)**2))]:
                max_metric_error = max(max_metric_error, abs(float(value)-row[key]))
                assert abs(float(value)-row[key]) < 1e-7
    for selection in frozen['selections']:
        seed, family, batch = selection['seed'], selection['family'], selection['batch']
        group = [r for r in timings if r['seed'] == seed]
        budget = next(r for r in group if r['name'] == 'cqs_r')[f'batch{batch}']['median_ms']
        assert budget == selection['reference_ms']
        candidates = [dict(name=r['name'], val_mse=r['val_mse'],
            latency_ms=r[f'batch{batch}']['median_ms']) for r in group
            if r['name'].startswith(family)]
        expected = select_under_budget(candidates, budget)
        assert expected == selection['constrained']
        assert min(candidates, key=lambda r: r['val_mse']) == selection['unconstrained']
    for name in names:
        values = [r['test_mae'] for r in rows if r['name'] == name]
        actual = summary['all_configurations'][name]
        assert abs(np.mean(values)-actual['mean']) < 1e-12
        assert abs(np.std(values, ddof=1)-actual['sd']) < 1e-12
    for s in summary['selected']:
        choices = [r for r in frozen['selections']
                   if r['batch'] == s['batch'] and r['family'] == s['family']]
        values = []
        for choice in choices:
            selected = choice[s['mode']]
            if selected:
                values.append(next(r['test_mae'] for r in rows if r['seed'] == choice['seed']
                                   and r['name'] == selected['name']))
        assert len(values) == s['eligible_datasets']
        if len(values) == 5:
            assert abs(np.mean(values)-s['test_mae']['mean']) < 1e-12
        else:
            assert s['test_mae'] is None
    result = dict(passed=True, full_parallel_fits=30, all_test_metrics_replayed=45,
        checkpoint_replays=45, full_configuration_test_predictions_retained=True,
        cpu_vs_gpu_checkpoint_prediction_max_error=max_prediction_error,
        cpu_replay_is_diagnostic_only=True,
        cuda_checkpoint_prediction_max_error=gpu['max_prediction_error'],
        cuda_checkpoint_replay_environment=gpu['torch'],
        saved_metric_max_error=max_metric_error, validation_argmins_reconstructed=True,
        latency_eligibility_reconstructed=True, dataset_sd_statistics_reconstructed=True,
        primary_budget='paired dataset CQS-R median, separate batch1 and400 contracts',
        missing_batch400_transformer_configurations='one dataset has no eligible Transformer; no five-dataset mean imputed',
        hardware_shots=0)
    write(evaluation/'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
