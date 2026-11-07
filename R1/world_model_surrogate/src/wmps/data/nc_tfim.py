"""TFIM simulation tasks with NC-style initial state, controls, and shadow labels."""
from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from scipy.sparse.linalg import eigsh


def shadow_terms(bits, recipes, edges):
    """Unbiased Pauli-shadow estimators, recipes 0=X,1=Y,2=Z; bits indexed by wire."""
    signs = 1.-2.*np.asarray(bits)
    recipes = np.asarray(recipes)
    xs = (3*signs*(recipes == 0)).mean(axis=0)
    zz = [np.mean(9*signs[:, i]*signs[:, j]*(recipes[:, i] == 2)*(recipes[:, j] == 2))
          for i,j in edges]
    return np.concatenate([xs, zz])


@dataclass
class TFIMTask:
    rows: int
    cols: int
    layers: int
    shared: bool
    hamiltonian_sign: int = -1

    def __post_init__(self):
        self.n = self.rows*self.cols
        if min(self.rows, self.cols, self.layers) < 1 or self.n < 2:
            raise ValueError("Invalid lattice")
        if self.hamiltonian_sign not in (-1,1):
            raise ValueError("Hamiltonian sign must be explicit")
        if self.rows == 1:
            self.edges = [(i,i+1) for parity in (0,1) for i in range(parity, self.n-1, 2)]
        else:
            self.edges = []
            for r in range(self.rows):
                for c in range(self.cols):
                    i = r*self.cols+c
                    if c+1 < self.cols:
                        self.edges.append((i,i+1))
                    if r+1 < self.rows:
                        self.edges.append((i,i+self.cols))
        self.block = self.n+len(self.edges)
        self.control_shape = (self.layers, 2 if self.shared else self.block)
        mapping = []
        for l in range(self.layers):
            mapping.extend(([2*l]*self.n+[2*l+1]*len(self.edges)) if self.shared
                           else range(l*self.block,(l+1)*self.block))
        self.gate_to_control = np.asarray(mapping)
        self.observables = []
        for i in range(self.n):
            self.observables.append(SparsePauliOp.from_sparse_list([("X",[i],1.)], num_qubits=self.n))
        for i,j in self.edges:
            self.observables.append(SparsePauliOp.from_sparse_list([("ZZ",[i,j],1.)], num_qubits=self.n))

    def circuit(self, controls):
        x = np.asarray(controls).reshape(self.control_shape)
        expanded = x.ravel()[self.gate_to_control].reshape(self.layers,self.block)
        q = QuantumCircuit(self.n)
        q.h(range(self.n))
        for row in expanded:
            for i in range(self.n):
                q.rx(float(row[i]),i)
            for e,(i,j) in enumerate(self.edges):
                q.rzz(float(row[self.n+e]),i,j)
        return q

    def state(self, controls):
        return Statevector(self.circuit(controls))

    def exact_terms(self, controls):
        state = self.state(controls)
        return np.array([state.expectation_value(o).real for o in self.observables])

    def coefficients(self, J, h):
        return self.hamiltonian_sign*np.array([h]*self.n+[J]*len(self.edges))

    def energy(self, controls, J, h):
        return float(self.exact_terms(controls) @ self.coefficients(J,h))

    def spectrum_bounds(self, J, h):
        terms = [("X",[i],self.hamiltonian_sign*h) for i in range(self.n)] + [("ZZ",list(e),self.hamiltonian_sign*J) for e in self.edges]
        matrix = SparsePauliOp.from_sparse_list(terms,num_qubits=self.n).to_matrix(sparse=True)
        if self.n <= 8:
            vals = np.linalg.eigvalsh(matrix.toarray())
            return float(vals[0]),float(vals[-1])
        return (float(eigsh(matrix,k=1,which="SA",return_eigenvectors=False)[0]),
                float(eigsh(matrix,k=1,which="LA",return_eigenvectors=False)[0]))

    def sample_shadow(self, controls, shots, rng):
        state = self.state(controls)
        recipes = rng.integers(0,3,size=(shots,self.n),dtype=np.int8)
        bits = np.empty_like(recipes)
        for recipe in np.unique(recipes,axis=0):
            ids = np.flatnonzero((recipes == recipe).all(axis=1))
            rotate = QuantumCircuit(self.n)
            for i,basis in enumerate(recipe):
                if basis == 1:
                    rotate.sdg(i)
                if basis != 2:
                    rotate.h(i)
            probabilities = state.evolve(rotate).probabilities()
            outcomes = rng.choice(len(probabilities),len(ids),p=probabilities/probabilities.sum())
            bits[ids] = (outcomes[:,None] >> np.arange(self.n)) & 1
        return bits,recipes

    def sample_grouped_terms(self, controls, shots, rng):
        """Two commuting measurement groups, `shots` per group, no shadow labels."""
        state = self.state(controls)
        probs_z = state.probabilities()
        q = QuantumCircuit(self.n)
        q.h(range(self.n))
        probs_x = state.evolve(q).probabilities()
        z = rng.multinomial(shots,probs_z/probs_z.sum())/shots
        x = rng.multinomial(shots,probs_x/probs_x.sum())/shots
        signs = 1-2*((np.arange(len(z))[:,None] >> np.arange(self.n)) & 1)
        return np.array([x @ signs[:,i] for i in range(self.n)] +
                        [z @ (signs[:,i]*signs[:,j]) for i,j in self.edges])

    def differentiable_energy(self, J, h):
        """PennyLane simulator oracle; adjoint differentiates shared parameters correctly."""
        import pennylane as qml
        obs = [qml.PauliX(i) for i in range(self.n)] + [qml.PauliZ(i)@qml.PauliZ(j) for i,j in self.edges]
        H = qml.Hamiltonian(self.coefficients(J,h),obs)
        dev = qml.device("default.qubit",wires=self.n)

        @qml.qnode(dev,interface="torch",diff_method="adjoint")
        def energy(x):
            for i in range(self.n):
                qml.Hadamard(i)
            for l in range(self.layers):
                for i in range(self.n):
                    qml.RX(x[l,0 if self.shared else i],wires=i)
                for e,(i,j) in enumerate(self.edges):
                    qml.IsingZZ(x[l,1 if self.shared else self.n+e],wires=[i,j])
            return qml.expval(H)
        return energy
