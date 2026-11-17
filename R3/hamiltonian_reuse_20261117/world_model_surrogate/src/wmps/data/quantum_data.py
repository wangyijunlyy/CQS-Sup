"""Quantum data generation for the WMPS project.

Builds 1D TFIM Trotter-ansatz circuits (as in Nature Commun. 17, 4731 (2026)),
runs them on a Qiskit simulator (statevector fast path when noise is off,
density-matrix path with a Pauli/depolarizing noise model when noise is on),
and returns the expectation values used as training labels.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector


def build_tfim_hamiltonian(N: int, J: float, h: float) -> SparsePauliOp:
    """Return the 1D TFIM Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    terms = []
    for i in range(N - 1):
        label = "I" * i + "ZZ" + "I" * (N - i - 2)
        terms.append((label, -J))
    for i in range(N):
        label = "I" * i + "X" + "I" * (N - i - 1)
        terms.append((label, -h))
    return SparsePauliOp.from_list(terms)


def build_circuit(x_layer: np.ndarray, N: int) -> QuantumCircuit:
    """Build one Trotter layer of the TFIM ansatz.

    U(x_l) = prod_i RX_i(x_{l,i}) * prod_i RZZ_{i,i+1}(x_{l,N+i}),
    with initial state |+>^{otimes N}. RZZ is decomposed exactly as
    CNOT(a,b) RZ(theta)_b CNOT(a,b) so that the noise model only needs
    'rx' and 'cx' gates.
    """
    d_layer = 2 * N - 1
    assert x_layer.shape[0] == d_layer, f"expected {d_layer} params per layer, got {x_layer.shape[0]}"
    circ = QuantumCircuit(N)
    circ.h(range(N))  # |+> initial state
    for i in range(N):
        circ.rx(x_layer[i], i)
    for i in range(N - 1):
        circ.cx(i, i + 1)
        circ.rz(x_layer[N + i], i + 1)
        circ.cx(i, i + 1)
    return circ


def build_noise_model(noise_level: float, pauli_weights: Optional[list] = None):
    """Build an Aer noise model:
    - single-qubit Pauli channel on 'rx' with total error probability p;
    - depolarizing channel on 'cx' with probability p (RZZ uses two CX gates).
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

    p = float(noise_level)
    if pauli_weights is None:
        pauli_weights = [0.45, 0.45, 0.1]
    w = np.asarray(pauli_weights, dtype=float)
    w = w / w.sum() * p
    err1 = pauli_error(
        [("X", w[0]), ("Y", w[1]), ("Z", w[2]), ("I", 1.0 - w.sum())]
    )
    err2 = depolarizing_error(p, 2)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(err1, ["rx"])
    nm.add_all_qubit_quantum_error(err2, ["cx"])
    return nm


def _marginals_from_amplitudes(psi: np.ndarray, N: int):
    """Single-qubit <Z_i>, <X_i> and two-qubit <Z_i Z_{i+1}> from a statevector."""
    pt = (np.abs(psi) ** 2).reshape([2] * N)
    z = np.zeros(N)
    x = np.zeros(N)
    psit = psi.reshape([2] * N)
    for i in range(N):
        p2 = np.moveaxis(pt, i, -1)
        z[i] = p2[..., 0].sum() - p2[..., 1].sum()
        s2 = np.moveaxis(psit, i, -1)
        x[i] = 2.0 * np.real(np.sum(np.conj(s2[..., 0]) * s2[..., 1]))
    zz = np.zeros(N - 1)
    for i in range(N - 1):
        p2 = np.moveaxis(pt, (i, i + 1), (-2, -1))
        zz[i] = (p2[..., 0, 0].sum() + p2[..., 1, 1].sum()
                 - p2[..., 0, 1].sum() - p2[..., 1, 0].sum())
    return z, zz, x


def _marginals_from_dm(dm_data: np.ndarray, N: int):
    """Same marginals from a density matrix (noisy case)."""
    p = np.real(np.diag(dm_data)).reshape([2] * N)
    z = np.zeros(N)
    for i in range(N):
        p2 = np.moveaxis(p, i, -1)
        z[i] = p2[..., 0].sum() - p2[..., 1].sum()
    zz = np.zeros(N - 1)
    for i in range(N - 1):
        p2 = np.moveaxis(p, (i, i + 1), (-2, -1))
        zz[i] = (p2[..., 0, 0].sum() + p2[..., 1, 1].sum()
                 - p2[..., 0, 1].sum() - p2[..., 1, 0].sum())
    x = np.zeros(N)
    dmr = dm_data.reshape([2] * (2 * N))
    for i in range(N):
        d2 = np.moveaxis(dmr, (i, N + i), (-2, -1))
        x[i] = 2.0 * np.real(np.sum(d2[..., 0, 1]))
    return z, zz, x


def generate_dataset(
    n_samples: int,
    N: int,
    L: int,
    J: float,
    h: float,
    noise_level: float = 0.0,
    pauli_weights: Optional[list] = None,
    x_range: float = np.pi,
    seed: int = 42,
    include_local_z: bool = False,
    shadow_T: int = 0,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
):
    """Sample x ~ Unif[-pi, pi]^d and compute y = <H_TFIM> (and optionally <Z_i>).

    If shadow_T > 0, also compute classical-shadow estimates of <H_TFIM>
    (T random Pauli snapshots per sample) as the h_cs training labels.

    Returns:
        X: np.ndarray [n, L, 2N-1]
        y: np.ndarray [n, m] where m = 1 (+N if include_local_z)
        y_shadow: np.ndarray [n, 1] only if shadow_T > 0
    """
    rng = np.random.default_rng(seed)
    d_layer = 2 * N - 1
    X = rng.uniform(-x_range, x_range, size=(n_samples, L, d_layer)).astype(np.float64)

    obs_energy = build_tfim_hamiltonian(N, J, h)
    obs_local = []
    if include_local_z:
        for i in range(N):
            obs_local.append(SparsePauliOp.from_list([("I" * i + "Z" + "I" * (N - i - 1), 1.0)]))

    sim = None
    if noise_level > 0:
        from qiskit_aer import AerSimulator

        noise_model = build_noise_model(noise_level, pauli_weights)
        sim = AerSimulator(method="density_matrix", noise_model=noise_model)

    m = 1 + (N if include_local_z else 0)
    y = np.zeros((n_samples, m), dtype=np.float64)
    y_shadow = np.zeros((n_samples, 1), dtype=np.float64) if shadow_T > 0 else None

    t0 = time.time()
    for i in range(n_samples):
        circ = build_circuit(X[i, 0], N)
        for layer in range(1, L):
            circ = circ.compose(build_circuit(X[i, layer], N))
        if sim is None:
            sv = Statevector(circ)
            y[i, 0] = float(np.real(sv.expectation_value(obs_energy)))
            for j, ob in enumerate(obs_local):
                y[i, 1 + j] = float(np.real(sv.expectation_value(ob)))
            if shadow_T > 0:
                z, zz, x = _marginals_from_amplitudes(sv.data, N)
        else:
            circ.save_density_matrix()
            result = sim.run(circ, shots=None).result()
            dm = DensityMatrix(result.data(0)["density_matrix"])
            y[i, 0] = float(np.real(dm.expectation_value(obs_energy)))
            for j, ob in enumerate(obs_local):
                y[i, 1 + j] = float(np.real(dm.expectation_value(ob)))
            if shadow_T > 0:
                z, zz, x = _marginals_from_dm(dm.data, N)
        if shadow_T > 0:
            y_shadow[i, 0] = _energy_shadow_estimate(z, zz, x, N, J, h, shadow_T, rng)
        if verbose and (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  [data] {i+1}/{n_samples} ({el:.1f}s)")

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"tfim_N{N}_L{L}_J{J}_h{h}_noise{noise_level}_xr{x_range}_n{n_samples}_seed{seed}_z{int(include_local_z)}_s{shadow_T}"
        if y_shadow is not None:
            np.savez_compressed(os.path.join(cache_dir, key + ".npz"), X=X, y=y, y_shadow=y_shadow)
        else:
            np.savez_compressed(os.path.join(cache_dir, key + ".npz"), X=X, y=y)
    return (X, y) if y_shadow is None else (X, y, y_shadow)


def _energy_shadow_estimate(z, zz, x, N, J, h, T, rng) -> float:
    """Pauli classical-shadow estimate of <H_TFIM> from T random snapshots.

    Uses the exact single-/two-qubit marginals to sample snapshot outcomes:
    - X_i: single-qubit shadow, factor 3, P(+1) = (1 + <X_i>)/2;
    - Z_i Z_{i+1}: two-qubit shadow, factor 9, joint P from <Z_i>,<Z_{i+1}>,<Z_i Z_{i+1}>.
    """
    est_zz = np.zeros(N - 1)
    est_x = np.zeros(N)
    for _ in range(T):
        basis = rng.integers(0, 3, size=N)
        for i in range(N - 1):
            if basis[i] == 2 and basis[i + 1] == 2:
                zi, zj, zij = z[i], z[i + 1], zz[i]
                p_pp = (1.0 + zi + zj + zij) / 4.0
                p_pm = (1.0 + zi - zj - zij) / 4.0
                p_mp = (1.0 - zi + zj - zij) / 4.0
                u = rng.random()
                if u < p_pp:
                    s = 1.0
                elif u < p_pp + p_pm:
                    s = -1.0
                elif u < p_pp + p_pm + p_mp:
                    s = -1.0
                else:
                    s = 1.0
                est_zz[i] += 9.0 * s
        for i in range(N):
            if basis[i] == 1:  # X basis
                s = 1.0 if rng.random() < (1.0 + x[i]) / 2.0 else -1.0
                est_x[i] += 3.0 * s
    est_zz /= T
    est_x /= T
    return float(-J * est_zz.sum() - h * est_x.sum())


def load_or_generate(
    n_samples: int,
    N: int,
    L: int,
    J: float,
    h: float,
    noise_level: float = 0.0,
    pauli_weights: Optional[list] = None,
    x_range: float = np.pi,
    seed: int = 42,
    include_local_z: bool = False,
    shadow_T: int = 0,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
):
    """Generate with caching; returns (X, y)."""
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"tfim_N{N}_L{L}_J{J}_h{h}_noise{noise_level}_xr{x_range}_n{n_samples}_seed{seed}_z{int(include_local_z)}_s{shadow_T}"
        path = os.path.join(cache_dir, key + ".npz")
        if os.path.exists(path):
            d = np.load(path)
            if verbose:
                print(f"[data] loaded cached dataset {os.path.basename(path)}")
            if shadow_T > 0:
                return d["X"], d["y"], d["y_shadow"]
            return d["X"], d["y"]
    return generate_dataset(
        n_samples=n_samples,
        N=N,
        L=L,
        J=J,
        h=h,
        noise_level=noise_level,
        pauli_weights=pauli_weights,
        x_range=x_range,
        seed=seed,
        include_local_z=include_local_z,
        shadow_T=shadow_T,
        cache_dir=cache_dir,
        verbose=verbose,
    )
