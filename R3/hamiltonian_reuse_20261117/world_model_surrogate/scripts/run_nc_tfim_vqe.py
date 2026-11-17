"""Matched simulator TFIM prediction, VQE pretraining, and shadow-data reuse.

No IBM access. New datasets only. h_cs/h_qs identify SI experimental regression
profiles, not the old Trig. ridge or the theoretical h_cs kernel estimator.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.wmps.data.nc_tfim import TFIMTask, shadow_terms
from src.wmps.models.nc_baselines import TrigFeatureMap, frequency_vectors, fit_nc_regression
from src.wmps.models.world_model import WorldModel
from src.wmps.train import metrics
from run_fspt8_fair import fit


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)


def initialize_run(out,protocol):
    path = out/"protocol.json"
    protocol = json.loads(json.dumps(protocol,allow_nan=False))
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Existing run uses a different protocol; choose a fresh output root")
    write_json(path,protocol)


def prepare(task, case, seed, out):
    path = out/"dataset.npz"
    if path.exists():
        with np.load(path) as a:
            return a["X"],a["exact_terms"],a["measured_terms"]
    rng = np.random.default_rng(seed)
    measured_count = case["train"]+case["val"]
    count = measured_count+case["test"]
    x = rng.uniform(-np.pi,np.pi,size=(count,*task.control_shape))
    exact = np.empty((count,task.block))
    measured = np.empty((measured_count,task.block))
    bits = np.empty((measured_count,case["shots"],task.n),dtype=np.int8) if not task.shared else None
    recipes = np.empty_like(bits) if bits is not None else None
    for i in range(count):
        exact[i] = task.exact_terms(x[i])
        if i < measured_count:
            if task.shared:
                measured[i] = task.sample_grouped_terms(x[i],case["shots"],rng)
            else:
                bits[i],recipes[i] = task.sample_shadow(x[i],case["shots"],rng)
                measured[i] = shadow_terms(bits[i],recipes[i],task.edges)
        if (i+1)%200 == 0:
            print(f"{case['id']} seed={seed} data {i+1}/{count}",flush=True)
    if bits is not None:
        np.savez_compressed(out/"shadow_records.npz",bits=bits,recipes=recipes)
    np.savez_compressed(path,X=x,exact_terms=exact,measured_terms=measured)
    return x,exact,measured


def optimize(predict, initial, config, exact_energy):
    """All restarts retained; true energies logged but never used for selection."""
    x = torch.tensor(initial,dtype=torch.float64,requires_grad=True)
    opt = torch.optim.Adam([x],lr=config["lr"],foreach=False)
    trajectory, losses, exact_steps, exact_values = [],[],[],[]
    start = time.perf_counter()
    for step in range(config["steps"]+1):
        opt.zero_grad()
        pred = predict(x).reshape(len(initial))
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Nonfinite VQE surrogate prediction")
        trajectory.append(x.detach().numpy().copy())
        losses.append(pred.detach().numpy())
        if step%config["exact_curve_every"] == 0 or step == config["steps"]:
            exact_steps.append(step)
            exact_values.append([exact_energy(z) for z in trajectory[-1]])
        if step == config["steps"]:
            break
        pred.sum().backward()
        if not torch.isfinite(x.grad).all():
            raise FloatingPointError("Nonfinite VQE gradient")
        opt.step()
        with torch.no_grad():
            x.clamp_(-config["clip"],config["clip"])
    return dict(parameters=np.array(trajectory),predictions=np.array(losses),
                exact_steps=np.array(exact_steps),exact_energy=np.array(exact_values),
                seconds=np.array(time.perf_counter()-start))


def run(config, case, seed, root, device, profile="si", dataset_root=None):
    out = root/f"{case['id']}_seed{seed}"
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sign = 1 if profile == "code" and case["shared"] else -1
    task = TFIMTask(case["rows"],case["cols"],case["layers"],case["shared"],sign)
    baseline = "h_qs" if task.shared else "h_cs"
    alpha = (0.1 if task.shared else 10.) if profile == "code" else 1.
    count = None if profile == "code" else case["feature_count"]
    protocol = dict(case=case,seed=seed,config=config["version"],baseline=baseline,
        source="NC experimental regression; new noiseless simulator with finite-shot labels",
        profile=profile,hamiltonian_sign=sign,actual_alpha=alpha,feature_budget=count,
        measured_training_and_validation_circuits=case["train"]+case["val"],
        shots_per_control=case["shots"]*(2 if task.shared else 1),
        h=-.5,initial_state="plus",hadamards="once only",edges=task.edges,
        baseline_fit=config["baseline"],redq=config["redq"],vqe=config["vqe"],
        public_code_setting_difference="SI alpha=1; code alpha=10 (h_cs),0.1 (h_qs); code h_qs uses +J ZZ +h X",
        model_selection="NC training EMA; RED-Q validation measured MSE",
        data_reuse="same stored measurements; both classical models refit per Hamiltonian",
        quantum_hardware_access=False)
    initialize_run(out,protocol)
    if (out/"results.json").exists():
        print(f"Already complete {out.name}",flush=True)
        return
    data_dir = out if dataset_root is None else dataset_root/f"{case['id']}_seed{seed}"
    data_dir.mkdir(parents=True,exist_ok=True)
    data_protocol = {k:case[k] for k in ("rows","cols","layers","shared","train","val","test","shots")}
    if data_dir != out:
        initialize_run(data_dir,dict(seed=seed,**data_protocol))
    x,exact,measured = prepare(task,case,seed,data_dir)
    nt,nv = case["train"],case["val"]
    tr,va,te = slice(0,nt),slice(nt,nt+nv),slice(nt+nv,None)
    w = frequency_vectors(len(task.gate_to_control),2,count,seed)
    np.save(out/"frequencies.npy",w)
    fmap = TrigFeatureMap(w,task.gate_to_control).to(device)
    with torch.no_grad():
        features = fmap(torch.tensor(x.reshape(len(x),-1),dtype=torch.float64,device=device))
    initial_path = ROOT/"results_nc_reference"/f"initial_f{len(w)}.npy"
    initial_weights = np.load(initial_path)
    coefficients = np.stack([task.coefficients(J,-.5) for J in case["couplings"]],axis=1)
    true_y, measured_y = exact@coefficients,measured@coefficients
    baseline_weights,detail = fit_nc_regression(features[tr],measured_y[tr],initial_weights,
        alpha=alpha,epochs=config["baseline"]["fit_epochs"],
        lr=config["baseline"]["fit_lr"])
    np.save(out/"baseline_weights.npy",baseline_weights)
    baseline_predictions = (features[te]@torch.tensor(baseline_weights,device=device)).cpu().numpy()
    np.savez_compressed(out/"prediction_targets.npz",exact=true_y[te],baseline=baseline_predictions)
    del features
    fmap = fmap.cpu()
    starts = np.random.default_rng(seed+10000).uniform(-np.pi,np.pi,
        size=(config["vqe"]["restarts"],*task.control_shape))
    np.save(out/"initial_parameters.npy",starts)
    records = []
    for j,J in enumerate(case["couplings"]):
        folder = out/f"J{J:+.1f}"
        folder.mkdir(exist_ok=True)
        if (folder/"results.json").exists():
            records.append(json.loads((folder/"results.json").read_text()))
            continue
        width = config["redq"]["width"]
        model,redq_detail = fit(lambda: WorldModel(in_dim=task.control_shape[1],out_dim=1,
            encoder_hidden=width,rssm_hidden=width,predictor_hidden=width,stochastic=False,dropout=0.),
            x[tr],measured_y[tr,j:j+1],x[va],measured_y[va,j:j+1],seed,device,folder,"redq",
            config["redq"]["epochs"])
        model.eval().cpu()
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            rp = model(torch.tensor(x[te],dtype=torch.float32))[0].numpy()
        np.save(folder/"pred_redq.npy",rp)
        bw = torch.tensor(baseline_weights[:,j],dtype=torch.float64)
        nc_predict = lambda z: fmap(z.reshape(len(z),-1))@bw
        redq_predict = lambda z: model(z.float())[0][:,0].double()
        oracle = task.differentiable_energy(J,-.5)
        oracle_predict = lambda z: torch.stack([oracle(a) for a in z])
        truth = lambda z: task.energy(z,J,-.5)
        # Cross-library check before optimizing shared or independent parameters.
        check = oracle(torch.tensor(starts[0],dtype=torch.float64)).detach().item()
        np.testing.assert_allclose(check,truth(starts[0]),atol=1e-10)
        lower,upper = task.spectrum_bounds(J,-.5)
        result = dict(J=J,h=-.5,N=task.n,features=len(w),baseline=baseline,seed=seed,
            prediction={baseline:metrics(true_y[te,j:j+1],baseline_predictions[:,j:j+1]),
                        "redq":metrics(true_y[te,j:j+1],rp)},
            baseline_training=detail[j],redq_training=redq_detail,spectrum=[lower,upper],vqe={})
        for name,predict in ((baseline,nc_predict),("redq",redq_predict),("exact_vqe",oracle_predict)):
            print(f"{out.name} J={J} optimizing {name}",flush=True)
            trace = optimize(predict,starts,config["vqe"],truth)
            np.savez_compressed(folder/f"vqe_{name}.npz",**trace)
            final = trace["exact_energy"][-1]
            result["vqe"][name] = dict(initial_mean=float(trace["exact_energy"][0].mean()),
                final_mean=float(final.mean()),restart_std=float(final.std(ddof=1)),
                energy_error_per_spin=float(((final-lower)/task.n).mean()),
                normalized_deviation=float(((final-lower)/(upper-lower)).mean()),
                surrogate_reported_final=float(trace["predictions"][-1].mean()),
                restarts=len(final),seconds=float(trace["seconds"]))
        write_json(folder/"results.json",result)
        records.append(result)
        print(json.dumps(dict(case=case["id"],seed=seed,J=J,prediction=result["prediction"],vqe=result["vqe"])),flush=True)
    write_json(out/"results.json",dict(completed=True,protocol=protocol,records=records))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,default=ROOT/"configs/nc_tfim_vqe.json")
    ap.add_argument("--case",required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--profile",choices=("si","code"),default="si")
    ap.add_argument("--dataset-root",type=Path)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    case = next(c for c in config["cases"] if c["id"] == args.case)
    if args.seed not in config["seeds"]:
        raise ValueError("Unregistered seed")
    run(config,case,args.seed,args.root,args.device,args.profile,args.dataset_root)
