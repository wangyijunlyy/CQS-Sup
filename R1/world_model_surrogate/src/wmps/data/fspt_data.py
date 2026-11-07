"""FSPT multi-period magnetization data with classical-shadow observations.

Implements the time-periodic Hamiltonian from the paper (Eq. 10):
    H_tp(t) = H_tp,1 = (pi/2 - delta) sum_i X_i,         0 <= t < T_1
              H_tp,2 = - sum_i J_i Z_{i-1} X_i Z_{i+1},  T_1 <= t < 2T_1
with open-boundary convention (Z_0 = Z_{N+1} = I).

The circuit is U(x) = (U_2(J) U_1(delta))^k for k driving periods.
At each period the true local magnetizations <Z_i(k)> are computed exactly
(statevector) and a Pauli-based classical shadow is simulated with T random
snapshots, yielding the noisy observation vector o_k used as world-model input.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def _z_obs(N: int, i: int) -> SparsePauliOp:
    return SparsePauliOp.from_list([("I" * i + "Z" + "I" * (N - i - 1), 1.0)])


def _apply_pauli_rotation(circ: QuantumCircuit, terms, theta: float) -> None:
    """Apply exp(-i*theta/2 * prod P) for P in {'X','Z'} on given qubits."""
    for q, p in terms:
        if p == "X":
            circ.h(q)
    target = terms[-1][0]
    for q, _ in terms[:-1]:
        circ.cx(q, target)
    circ.rz(theta, target)
    for q, _ in reversed(terms[:-1]):
        circ.cx(q, target)
    for q, p in reversed(terms):
        if p == "X":
            circ.h(q)


def build_period_block(N: int, delta: float, J: np.ndarray) -> QuantumCircuit:
    """One driving period U(x) = U_2(J) U_1(delta)."""
    circ = QuantumCircuit(N)
    # U_1: exp(-i (pi/2 - delta) sum X_i) = prod_i RX(pi - 2*delta)
    for i in range(N):
        circ.rx(np.pi - 2.0 * delta, i)
    # U_2: exp(+i sum_i J_i Z_{i-1} X_i Z_{i+1}) = prod R_P(-2 J_i)
    for i in range(N):
        theta = -2.0 * J[i]
        if N == 1:
            circ.rz(theta, 0)
        elif i == 0:
            _apply_pauli_rotation(circ, [(0, "X"), (1, "Z")], theta)
        elif i == N - 1:
            _apply_pauli_rotation(circ, [(N - 2, "Z"), (N - 1, "X")], theta)
        else:
            _apply_pauli_rotation(circ, [(i - 1, "Z"), (i, "X"), (i + 1, "Z")], theta)
    return circ


def _shadow_estimates(true_z: np.ndarray, T: int, rng) -> np.ndarray:
    """Pauli classical-shadow estimates of <Z_i> from T random snapshots.

    Each snapshot picks a uniform random basis {X,Y,Z} per qubit; only
    Z-basis snapshots contribute to the <Z_i> estimate. The outcome marginal
    P(+1) = (1 + <Z_i>)/2 is exact for any state.
    """
    N = len(true_z)
    if T <= 0:
        return np.zeros(N)
    est = np.zeros(N)
    for _ in range(T):
        basis = rng.integers(0, 3, size=N)
        for i in range(N):
            if basis[i] == 2:  # Z basis
                m = 1.0 if rng.random() < (1.0 + true_z[i]) / 2.0 else -1.0
                est[i] += 3.0 * m
    return est / T


def generate_fspt_dataset(
    n_samples: int,
    N: int,
    K: int,
    T_shadow: int,
    delta_range=(0.0, 2.0),
    J_range=(0.0, 1.0),
    seed: int = 42,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
):
    """Sample x = (delta, J) and compute shadow observations + true magnetizations.

    Returns:
        X:   np.ndarray [n, N+1]           (delta, J_1..J_N)
        obs: np.ndarray [n, K, N]          classical-shadow estimates of <Z_i(k)>
        y:   np.ndarray [n, K, N]          true <Z_i(k)>
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, N + 1))
    obs = np.zeros((n_samples, K, N))
    y = np.zeros((n_samples, K, N))
    t0 = time.time()
    for s in range(n_samples):
        delta = rng.uniform(*delta_range)
        J = rng.uniform(*J_range, size=N)
        X[s] = np.concatenate([[delta], J])
        block = build_period_block(N, delta, J)
        circ = QuantumCircuit(N)
        for k in range(1, K + 1):
            circ = circ.compose(block)
            sv = Statevector(circ)
            z_true = np.array(
                [float(np.real(sv.expectation_value(_z_obs(N, i)))) for i in range(N)]
            )
            y[s, k - 1] = z_true
            obs[s, k - 1] = _shadow_estimates(z_true, T_shadow, rng)
        if verbose and (s + 1) % 100 == 0:
            print(f"  [fspt] {s+1}/{n_samples} ({time.time()-t0:.1f}s)")

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"fspt_N{N}_K{K}_T{T_shadow}_n{n_samples}_seed{seed}"
        np.savez_compressed(os.path.join(cache_dir, key + ".npz"), X=X, obs=obs, y=y)
    return X, obs, y


def load_or_generate_fspt(
    n_samples: int,
    N: int,
    K: int,
    T_shadow: int,
    delta_range=(0.0, 2.0),
    J_range=(0.0, 1.0),
    seed: int = 42,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"fspt_N{N}_K{K}_T{T_shadow}_n{n_samples}_seed{seed}"
        path = os.path.join(cache_dir, key + ".npz")
        if os.path.exists(path):
            d = np.load(path)
            if verbose:
                print(f"[fspt] loaded cached dataset {os.path.basename(path)}")
            return d["X"], d["obs"], d["y"]
    return generate_fspt_dataset(
        n_samples=n_samples, N=N, K=K, T_shadow=T_shadow,
        delta_range=delta_range, J_range=J_range, seed=seed,
        cache_dir=cache_dir, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# GPU data generation for the paper-aligned experiment (N=20, K=40).
# Runs in the dedicated `aer-gpu` environment (qiskit-aer GPU backend).
# ---------------------------------------------------------------------------

def _marginals_from_states(psis, N, K):
    """<Z_i> for PHYSICAL qubit i from saved statevectors.

    qiskit basis |q_{N-1}...q_0> is little-endian (qubit 0 = LSB of the flat
    index). Reshaping the flat index to [2]*N puts bit j on axis N-1-j, so
    physical qubit i (bit i) lives on axis N-1-i.
    """
    z = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        p = (np.abs(psis[k]) ** 2).astype(np.float32).reshape([2] * N)
        for i in range(N):
            z[k, i] = 2.0 * float(np.take(p, 0, axis=N - 1 - i).sum()) - 1.0
    return z


def generate_fspt_trajectories_gpu(
    params: np.ndarray,
    N: int,
    K: int,
    T_shadow: int,
    seed: int = 0,
    device: str = "GPU",
    verbose: bool = True,
):
    """Compute exact magnetizations (and shadow obs) for given parameter rows.

    params: [m, N+1] with columns (delta, J_1..J_N). Simulates each sample in a
    single Aer GPU run with save_statevector at every driving period.
    Returns (obs [m, K, N], y [m, K, N]).
    """
    from qiskit_aer import AerSimulator

    rng = np.random.default_rng(seed)
    m = params.shape[0]
    obs = np.zeros((m, K, N), dtype=np.float32)
    y = np.zeros((m, K, N), dtype=np.float32)
    sim = AerSimulator(method="statevector", device=device)
    t0 = time.time()
    for s in range(m):
        delta, J = params[s, 0], params[s, 1:]
        block = build_period_block(N, float(delta), J)
        circ = QuantumCircuit(N)
        for k in range(1, K + 1):
            circ = circ.compose(block)
            circ.save_statevector(label=f"sv{k}")
        res = sim.run(circ).result()
        psis = [res.data(0)[f"sv{k}"].data for k in range(1, K + 1)]
        z = _marginals_from_states(psis, N, K)
        y[s] = z
        for k in range(K):
            obs[s, k] = _shadow_estimates(z[k], T_shadow, rng)
        if verbose and (s + 1) % 50 == 0:
            print(f"  [fspt-gpu] {s+1}/{m} ({time.time()-t0:.1f}s)")
    return obs, y


def generate_fspt_dataset_gpu(
    n_samples: int,
    N: int,
    K: int,
    T_shadow: int,
    delta_range=(0.0, 2.0),
    J_range=(0.0, 1.0),
    seed: int = 42,
    cache_dir: Optional[str] = None,
    device: str = "GPU",
    verbose: bool = True,
):
    """Paper-aligned random dataset (x -> full magnetization trajectory)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, N + 1), dtype=np.float32)
    X[:, 0] = rng.uniform(*delta_range, size=n_samples)
    X[:, 1:] = rng.uniform(*J_range, size=(n_samples, N))
    obs, y = generate_fspt_trajectories_gpu(
        X, N, K, T_shadow, seed=seed, device=device, verbose=verbose
    )
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"fsptN{N}_K{K}_T{T_shadow}_n{n_samples}_seed{seed}"
        np.savez_compressed(os.path.join(cache_dir, key + ".npz"), X=X, obs=obs, y=y)
    return X, obs, y


def load_or_generate_fspt_gpu(
    n_samples: int,
    N: int,
    K: int,
    T_shadow: int,
    delta_range=(0.0, 2.0),
    J_range=(0.0, 1.0),
    seed: int = 42,
    cache_dir: Optional[str] = None,
    device: str = "GPU",
    verbose: bool = True,
):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"fsptN{N}_K{K}_T{T_shadow}_n{n_samples}_seed{seed}"
        path = os.path.join(cache_dir, key + ".npz")
        if os.path.exists(path):
            d = np.load(path)
            if verbose:
                print(f"[fspt-gpu] loaded cached {os.path.basename(path)}")
            return d["X"], d["obs"], d["y"]
    return generate_fspt_dataset_gpu(
        n_samples=n_samples, N=N, K=K, T_shadow=T_shadow,
        delta_range=delta_range, J_range=J_range, seed=seed,
        cache_dir=cache_dir, device=device, verbose=verbose,
    )


def generate_fspt_phase_grid_gpu(
    N: int, K: int, n_delta: int, n_J: int,
    delta_range=(0.0, 2.0), J_range=(0.0, 1.0), seed: int = 7,
    device: str = "GPU", cache_dir: Optional[str] = None, verbose: bool = True,
):
    """Structured (delta grid x J samples) reference grid for phase identification."""
    rng = np.random.default_rng(seed)
    deltas = np.linspace(*delta_range, n_delta)
    m = n_delta * n_J
    params = np.zeros((m, N + 1), dtype=np.float32)
    for d_i, delta in enumerate(deltas):
        for j_i in range(n_J):
            params[d_i * n_J + j_i, 0] = delta
            params[d_i * n_J + j_i, 1:] = rng.uniform(*J_range, size=N)
    obs, y = generate_fspt_trajectories_gpu(
        params, N, K, T_shadow=0, seed=seed, device=device, verbose=verbose
    )
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"fsptgrid_N{N}_K{K}_d{n_delta}_j{n_J}_seed{seed}"
        np.savez_compressed(os.path.join(cache_dir, key + ".npz"),
                            params=params, y=y)
    return params, y
