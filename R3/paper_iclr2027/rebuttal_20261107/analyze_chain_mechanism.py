"""Post-hoc mechanism analysis of saved searches; no fitting or new acquisition."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chain_depth_recovery import load_models, predictors, SEEDS, COUPLINGS
from run_evidence import ROOT, TFIMTask, ChainExact


def main():
    torch.set_num_threads(2)
    source = HERE / 'evidence/chain_depth_recovery'
    dest = HERE / 'evidence/chain_mechanism_20261108'
    dest.mkdir(exist_ok=True)
    task = TFIMTask(1, 6, 2, False, -1)
    exact = ChainExact(task)
    rows = []
    with torch.inference_mode():
        for seed in SEEDS:
            models = load_models(seed, task)
            with np.load(ROOT / f'results_nc_tfim_vqe_independent/datasets/chain6_l2_seed{seed}/dataset.npz') as data:
                test_x = torch.tensor(data['X'][2400:], dtype=torch.float64)
            for coupling in COUPLINGS:
                funcs = predictors(models, task, seed, coupling)
                for method in ('product', 'product_valrank', 'tuned_nc'):
                    predict = funcs[method]
                    test_error = (exact(test_x, coupling, -1) - predict(test_x, 0.)).numpy()
                    for arm in ('uniform_joint', 'shallow_joint', 'shallow_staged'):
                        path = source / f'seed{seed}/{method}_{arm}_J{coupling:+.1f}.npz'
                        with np.load(path) as saved:
                            trace = saved['parameters'][[0, 60, 100]]
                            stored_pred = saved['prediction'].copy()
                        x = torch.tensor(trace.reshape(-1, 2, 11), dtype=torch.float64)
                        truth = exact(x, coupling, -1).numpy().reshape(3, 20)
                        pred = predict(x, 0.).numpy().reshape(3, 20)
                        np.testing.assert_allclose(pred[-1], stored_pred, atol=1e-10, rtol=1e-10)
                        np.testing.assert_allclose(truth[-1], np.load(path.with_suffix('.exact.npy')), atol=1e-10, rtol=0)
                        error = truth - pred
                        for begin, label in ((0, 'full_search'), (1, 'last40_unsmoothed_updates')):
                            de = truth[-1] - truth[begin]
                            dp = pred[-1] - pred[begin]
                            derr = error[-1] - error[begin]
                            np.testing.assert_allclose(de, dp + derr, atol=1e-12, rtol=0)
                            rows.append(dict(seed=seed, J=coupling, method=method, arm=arm, interval=label,
                                start_true_mean=float(truth[begin].mean()), end_true_mean=float(truth[-1].mean()),
                                start_predicted_mean=float(pred[begin].mean()), end_predicted_mean=float(pred[-1].mean()),
                                mean_true_change=float(de.mean()), mean_prediction_change=float(dp.mean()),
                                mean_error_change=float(derr.mean()),
                                descending_prediction_rising_true_count=int(((dp < -1e-8) & (de > 1e-8)).sum()),
                                endpoint_signed_error=float(error[-1].mean()),
                                endpoint_energy_mae=float(abs(error[-1]).mean()),
                                random_test_signed_error=float(test_error.mean()),
                                random_test_energy_mae=float(abs(test_error).mean())))
            print('Replayed seed', seed, flush=True)
    local_rows = []
    saved_local_diagnostics = json.loads((source / 'local_diagnostics.json').read_text())
    with torch.inference_mode():
        for seed in SEEDS:
            models = load_models(seed, task)
            for coupling in COUPLINGS:
                predict = predictors(models, task, seed, coupling)['product']
                folder = source / f'local_calibration/seed{seed}_product_J{coupling:+.1f}'
                with np.load(folder / 'round2.npz') as saved:
                    candidates = torch.tensor(saved['candidate_parameters'], dtype=torch.float64)
                    corrected = saved['candidate_prediction'].copy()
                truth = exact(candidates, coupling, -1).numpy()
                prior = predict(candidates, 0.).numpy()
                replay = next(row for row in saved_local_diagnostics
                              if (row['seed'], row['J'], row['method']) == (seed, coupling, 'product'))
                np.testing.assert_allclose(abs(truth-corrected).mean(),
                                           replay['corrected_endpoint_energy_mae'], atol=1e-12, rtol=0)
                local_rows.append(dict(seed=seed, J=coupling,
                    original_prior_mae=float(abs(truth-prior).mean()),
                    corrected_mae=float(abs(truth-corrected).mean())))
    groups = []
    keys = ('J', 'method', 'arm', 'interval')
    for key in dict.fromkeys(tuple(row[k] for k in keys) for row in rows):
        group = [row for row in rows if tuple(row[k] for k in keys) == key]
        assert len(group) == 3
        result = dict(zip(keys, key))
        for metric in group[0]:
            if metric in (*keys, 'seed'):
                continue
            values = np.array([row[metric] for row in group])
            result[metric] = dict(mean=float(values.mean()), sd=float(values.std(ddof=1)), values=values.tolist())
        groups.append(result)
    (dest / 'per_run.json').write_text(json.dumps(rows, indent=2) + '\n')
    (dest / 'summary.json').write_text(json.dumps(groups, indent=2) + '\n')
    (dest / 'local_same_candidates.json').write_text(json.dumps(local_rows, indent=2) + '\n')
    get = lambda arm: next(row for row in groups if row['J'] == -.1 and row['method'] == 'product'
                           and row['arm'] == arm and row['interval'] == 'full_search')
    table_rows = [(get('shallow_joint'), 'start', 'Shallow solution with identity layer', '浅层解加恒等层'),
                  (get('shallow_joint'), 'end', 'After joint search, periodic wrapping', '周期回绕联合搜索后'),
                  (get('shallow_staged'), 'end', 'After staged search, periodic wrapping', '周期回绕分阶段搜索后')]
    for lang, folder in [('en', HERE.parent), ('zh', HERE.parent / 'chinese')]:
        header = ['Saved controls', 'Predicted energy', 'True energy'] if lang == 'en' else ['保存的控制参数', '预测能量', '真实能量']
        lines = [r'\begin{tabular}{lcc}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
        for row, point, en, zh in table_rows:
            cells = [en if lang == 'en' else zh]
            for kind in ('predicted', 'true'):
                metric = row[f'{point}_{kind}_mean']
                cells.append(f"${metric['mean']:.3f}\\pm{metric['sd']:.3f}$")
            lines.append(' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        (folder / 'tables/chain_mechanism.tex').write_text('\n'.join(lines) + '\n')
    check = dict(passed=True, saved_searches=len(rows)//2, intervals_per_search=2,
        exact_replay=True, endpoint_prediction_replay=True, error_decomposition_identity=True,
        local_correction_error_replay=True, local_comparisons=len(local_rows),
        new_fits=0, new_searches=0, new_measurements=0,
        scope='Post-hoc scoring of all saved arms and couplings; not a causal ablation of calibration versus acceptance.')
    (dest / 'verification.json').write_text(json.dumps(check, indent=2) + '\n')
    for group in groups:
        if group['J'] == -.1 and group['method'] == 'product':
            print(group['arm'], group['interval'], {k:round(group[k]['mean'], 6) for k in
                ('start_true_mean','end_true_mean','start_predicted_mean','end_predicted_mean',
                 'mean_true_change','mean_prediction_change','mean_error_change',
                 'descending_prediction_rising_true_count','endpoint_energy_mae','random_test_energy_mae')})
    for coupling in COUPLINGS:
        selected = [row for row in local_rows if row['J'] == coupling]
        print('Local same-proposal MAE', coupling,
              {key:round(float(np.mean([row[key] for row in selected])), 6)
               for key in ('original_prior_mae', 'corrected_mae')})
    print(json.dumps(check))


if __name__ == '__main__':
    main()
