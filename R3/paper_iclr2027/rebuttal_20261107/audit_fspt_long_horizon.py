"""Replay existing nested-budget predictions against validation-tuned NC.

No new training or model selection: report both stored budgets on original splits.
"""
from pathlib import Path
import csv
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'world_model_surrogate'


def main():
    with np.load(ROOT/'results_fspt_scaling/datasets/dataset_n8.npz') as data:
        targets = data['y']
    rows = []
    for seed in (42, 43, 44):
        folder = ROOT/f'results_fspt_diagnostics/n8_w128_seed{seed}'
        recorded = json.loads((folder/'results.json').read_text())
        with np.load(folder/'split.npz') as a, np.load(
                ROOT/f'results_fspt_scaling/n8_k24_seed{seed}/split.npz') as b:
            for key in ('train', 'val', 'test'):
                assert np.array_equal(a[key], b[key])
            truth = targets[a['test'], :24]
        assert recorded['protocol']['budgets'] == [300, 600]
        assert recorded['protocol']['selection'].startswith('minimum validation MSE')
        for cap in (300, 600):
            prediction = np.load(folder/f'pred_budget{cap}.npy')
            mae = float(np.abs(prediction.astype(float)-truth.astype(float)).mean())
            saved = recorded['budgets'][str(cap)]
            assert abs(mae-saved['metrics']['overall']['mae']) < 1e-7
            if cap == 300:
                baseline = np.load(ROOT/f'results_fspt_scaling/n8_k24_seed{seed}/pred_redq.npy')
                assert np.max(np.abs(prediction-baseline)) < 1e-6
            rows.append(dict(seed=seed, method=f'CQS-R cap{cap}', mae=mae,
                             first6_mae=float(np.abs(prediction[:, :6]-truth[:, :6]).mean()),
                             selected_epoch=saved['training']['best_epoch']))
        prediction = np.load(HERE/f'evidence/nc_tuned/fspt_n8_k24_seed{seed}/pred_per_output_alpha.npy')
        rows.append(dict(seed=seed, method='NC per-output tuned',
                         mae=float(np.abs(prediction.astype(float)-truth.astype(float)).mean()),
                         first6_mae=float(np.abs(prediction[:, :6]-truth[:, :6]).mean()),
                         selected_epoch=None))
    summary = {}
    for method in dict.fromkeys(r['method'] for r in rows):
        values = np.array([r['mae'] for r in rows if r['method'] == method])
        summary[method] = dict(mean=float(values.mean()), sample_sd=float(values.std(ddof=1)),
                               seed_values=values.tolist())
    extended = np.array(summary['CQS-R cap600']['seed_values'])
    nc = np.array(summary['NC per-output tuned']['seed_values'])
    summary['comparison'] = dict(
        nc_minus_extended_per_seed=(nc-extended).tolist(),
        relative_mae_reduction=1-float(extended.mean()/nc.mean()),
        scope='three paired split/initialization runs on one existing pool; additional neural training budget',
        checks='same splits; exact-target replay; cap300 baseline replay; validation-only stored checkpoint selection')
    out = HERE/'evidence'
    with (out/'fspt_long_horizon_budget_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/'fspt_long_horizon_budget_comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
