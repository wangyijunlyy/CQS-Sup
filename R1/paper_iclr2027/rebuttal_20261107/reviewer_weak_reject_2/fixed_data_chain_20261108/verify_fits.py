"""Rebuild every measured-validation decision from saved full checkpoints."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import json
import numpy as np
import torch
from run_tuned import HERE,ROOT,models
from fitting import energy_validation,write
from search import COUPLINGS
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.periodic_redq import ProductMomentREDQ


def main():
    torch.set_num_threads(2)
    task=TFIMTask(1,6,2,False,-1);coeff=np.stack([task.coefficients(j,-.5) for j in COUPLINGS])
    max_p=0.;max_f=0.;max_export=0.;full_fits=0;selections=[]
    for seed in (45,46,47):
        base=HERE/'tuned'/f'seed{seed}'
        assert (base/'all_fits_saved.json').exists()
        with np.load(ROOT/f'results_nc_tfim_vqe_independent/datasets/chain6_l2_seed{seed}/dataset.npz') as a:
            x,y=a['X'][2000:2400],a['measured_terms'][2000:2400]
        ps=json.loads((base/'product_selection.json').read_text())
        assert len(ps['candidates'])==31
        check=[]
        for row in ps['candidates']:
            model=ProductMomentREDQ(6,task.edges,2,False,row['rank'],1).float().eval()
            path=(ROOT/f'results_product_redq_independent/chain6_l2_seed{seed}/periodic_redq.pt') if row['candidate']=='historical' else base/'products'/row['candidate']/'model.pt'
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            with torch.no_grad():pred=model(torch.tensor(x,dtype=torch.float32))[0].numpy()
            score=energy_validation(pred,y,coeff)
            error=abs(score-row['validation_energy_mse']);max_p=max(max_p,error)
            assert error<2e-6*max(1.,abs(score)),(seed,row['candidate'],error)
            check.append((score,row['candidate']))
            if row['candidate']!='historical':full_fits+=1
        selected=min(ps['candidates'],key=lambda row:row['validation_energy_mse'])
        assert selected['candidate']==ps['selected']['candidate']
        # Near-ties are judged by the stored common-environment scores, not a different CPU backend.
        fs=json.loads((base/'fourier_selection.json').read_text())
        assert len(fs['candidates'])==16
        with np.load(base/'fourier_validation_predictions.npz') as a:predictions=a['predictions']
        for row,pred in zip(fs['candidates'],predictions):
            score=energy_validation(pred,y,coeff)
            error=abs(score-row['validation_energy_mse']);max_f=max(max_f,error)
            assert error<1e-10
        index=min(range(16),key=lambda i:fs['candidates'][i]['validation_energy_mse'])
        assert index==fs['selected_index']
        _,_,fourier=models(seed)
        with torch.no_grad():export=np.concatenate([fourier(torch.tensor(batch)).numpy() for batch in np.array_split(x,16)])
        error=float(abs(export-predictions[index]).max());max_export=max(max_export,error)
        assert error<1e-8,error
        selections.append(dict(seed=seed,product=ps['selected']['candidate'],fourier=fs['selected']))
    report=dict(passed=True,full_product_checkpoints_replayed=full_fits,historical_product_candidates_replayed=3,
        fourier_validation_candidates_replayed=48,selected_fourier_exports_replayed=3,
        max_product_validation_error=max_p,max_fourier_validation_error=max_f,max_fourier_export_error=max_export,
        validation_uses_measured_labels_only=True,one_shared_selection_for_six_couplings=True,selections=selections)
    write(HERE/'verification_fits.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
