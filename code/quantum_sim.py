"""
Minimal exact statevector simulator + structural gate/depth accounting.

Used to (a) compute REAL circuit-depth / gate-count numbers for the baseline
QRL-SHO-XGBoost encoder and the proposed Squire-IDS encoder from actual gate
lists (not hand-counted), and (b) produce exact expectation values <Z_q> for
both encoders on real feature vectors, with an optional finite-shot
(binomial) noise model to reproduce the baseline's 8192-shot estimation.

No external quantum SDK is used; qubit counts here (<=8) make dense
statevector simulation (2**n complex amplitudes) trivial.
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


GATE_FN = {"RX": rx, "RY": ry, "RZ": rz}


class Circuit:
    """Records a gate list (for depth/gate accounting) and can execute it
    on a statevector. Gate = (name, qubit_indices, param_or_None)."""

    def __init__(self, n_qubits):
        self.n = n_qubits
        self.gates = []  # list of (name, qubits tuple, theta or None)

    def add(self, name, qubits, theta=None):
        self.gates.append((name, tuple(qubits), theta))

    def one_qubit_gate_count(self):
        return sum(1 for g in self.gates if g[0] in GATE_FN)

    def two_qubit_gate_count(self):
        return sum(1 for g in self.gates if g[0] == "CNOT")

    def depth(self):
        """Standard 'critical path' depth: each qubit tracks the depth of
        the last layer that touched it; a gate's layer = 1 + max over its
        qubits' last-touched layer."""
        last = [0] * self.n
        for name, qubits, _theta in self.gates:
            layer = max(last[q] for q in qubits) + 1
            for q in qubits:
                last[q] = layer
        return max(last) if last else 0

    def statevector(self, init=None):
        n = self.n
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1.0
        psi = psi.reshape([2] * n)
        for name, qubits, theta in self.gates:
            if name in GATE_FN:
                (q,) = qubits
                U = GATE_FN[name](theta)
                psi = np.tensordot(U, psi, axes=([1], [q]))
                psi = np.moveaxis(psi, 0, q)
            elif name == "CNOT":
                c, t = qubits
                psi = np.moveaxis(psi, [c, t], [0, 1])
                out = psi.copy()
                # |1>_c |0>_t <-> |1>_c |1>_t swap on the target axis
                out[1, 0] = psi[1, 1]
                out[1, 1] = psi[1, 0]
                psi = np.moveaxis(out, [0, 1], [c, t])
            else:
                raise ValueError(name)
        return psi.reshape(-1)

    def z_expectations(self, init=None):
        psi = self.statevector(init)
        probs = np.abs(psi) ** 2
        n = self.n
        idx = np.arange(2 ** n)
        exps = np.zeros(n)
        for q in range(n):
            bit = (idx >> (n - 1 - q)) & 1  # qubit 0 = most significant axis
            exps[q] = probs[bit == 0].sum() - probs[bit == 1].sum()
        return exps


def baseline_circuit(features_8):
    """QRL-SHO-XGBoost encoder: 8 qubits, 1 feature/qubit angle (RY) encoding,
    ring CNOT entanglement (i,i+1 for i=1..7 plus closing 8->1), one
    variational layer of RZ-RY-RX per qubit (paper eq. 11-14, Table VI gate
    counts: 8 qubits -> depth 24, 72 1-qubit gates, 24 2-qubit gates)."""
    assert len(features_8) == 8
    rng = np.random.default_rng(1234)  # fixed random ansatz weights (paper does not specify trained values)
    circ = Circuit(8)
    for q in range(8):
        circ.add("RY", [q], float(features_8[q]))
    for q in range(7):
        circ.add("CNOT", [q, q + 1])
    circ.add("CNOT", [7, 0])  # ring closure
    phis = rng.uniform(0, 2 * np.pi, size=(8, 3))
    for q in range(8):
        circ.add("RZ", [q], float(phis[q, 0]))
        circ.add("RY", [q], float(phis[q, 1]))
        circ.add("RX", [q], float(phis[q, 2]))
    return circ


def squire_circuit(features_k):
    """Squire-IDS encoder: dense tri-axis angle encoding packs 3 features per
    qubit via sequential RX-RY-RZ data rotations, so n_qubits = ceil(k/3).
    Entanglement is a single brick-wall (linear nearest-neighbour, no ring
    closure) CNOT layer, and the trainable ansatz is a single RY-RZ layer
    (2 gates/qubit instead of baseline's 3), keeping depth ~constant as
    qubit count grows instead of scaling linearly with it."""
    k = len(features_k)
    n = int(np.ceil(k / 3))
    padded = list(features_k) + [0.0] * (3 * n - k)
    rng = np.random.default_rng(1234)
    circ = Circuit(n)
    for q in range(n):
        fx, fy, fz = padded[3 * q: 3 * q + 3]
        circ.add("RX", [q], float(fx))
        circ.add("RY", [q], float(fy))
        circ.add("RZ", [q], float(fz))
    for q in range(n - 1):
        circ.add("CNOT", [q, q + 1])
    phis = rng.uniform(0, 2 * np.pi, size=(n, 2))
    for q in range(n):
        circ.add("RY", [q], float(phis[q, 0]))
        circ.add("RZ", [q], float(phis[q, 1]))
    return circ


def _batched_rot(kind, thetas):
    c = np.cos(thetas / 2)
    s = np.sin(thetas / 2)
    U = np.zeros((len(thetas), 2, 2), dtype=complex)
    if kind == "RX":
        U[:, 0, 0] = c; U[:, 0, 1] = -1j * s
        U[:, 1, 0] = -1j * s; U[:, 1, 1] = c
    elif kind == "RY":
        U[:, 0, 0] = c; U[:, 0, 1] = -s
        U[:, 1, 0] = s; U[:, 1, 1] = c
    elif kind == "RZ":
        U[:, 0, 0] = np.exp(-1j * thetas / 2)
        U[:, 1, 1] = np.exp(1j * thetas / 2)
    else:
        raise ValueError(kind)
    return U


def _apply_1q_batched(psi, U_batch, qubit):
    psi = np.moveaxis(psi, qubit + 1, 1)
    psi = np.einsum("bij,bj...->bi...", U_batch, psi)
    psi = np.moveaxis(psi, 1, qubit + 1)
    return psi


def _apply_cnot_batched(psi, control, target):
    psi = np.moveaxis(psi, [control + 1, target + 1], [1, 2])
    out = psi.copy()
    out[:, 1, 0] = psi[:, 1, 1]
    out[:, 1, 1] = psi[:, 1, 0]
    return np.moveaxis(out, [1, 2], [control + 1, target + 1])


def _init_batched_psi(batch, n):
    psi = np.zeros((batch, 2 ** n), dtype=complex)
    psi[:, 0] = 1.0
    return psi.reshape((batch,) + (2,) * n)


def _batched_z_expectations(psi):
    batch = psi.shape[0]
    n = psi.ndim - 1
    flat = psi.reshape(batch, -1)
    probs = np.abs(flat) ** 2
    idx = np.arange(2 ** n)
    exps = np.zeros((batch, n))
    for q in range(n):
        bit = (idx >> (n - 1 - q)) & 1
        exps[:, q] = probs[:, bit == 0].sum(1) - probs[:, bit == 1].sum(1)
    return exps


def batched_baseline_embed(X8):
    """Vectorised equivalent of baseline_circuit(...).z_expectations() applied
    row-wise to X8 (n_samples, 8) features already scaled to [0, pi]."""
    batch = X8.shape[0]
    n = 8
    psi = _init_batched_psi(batch, n)
    for q in range(n):
        psi = _apply_1q_batched(psi, _batched_rot("RY", X8[:, q]), q)
    for q in range(n - 1):
        psi = _apply_cnot_batched(psi, q, q + 1)
    psi = _apply_cnot_batched(psi, n - 1, 0)
    rng = np.random.default_rng(1234)
    phis = rng.uniform(0, 2 * np.pi, size=(n, 3))
    for q in range(n):
        for kind, col in (("RZ", 0), ("RY", 1), ("RX", 2)):
            theta = np.full(batch, phis[q, col])
            psi = _apply_1q_batched(psi, _batched_rot(kind, theta), q)
    return _batched_z_expectations(psi)


def batched_squire_embed(Xk):
    """Vectorised equivalent of squire_circuit(...).z_expectations() applied
    row-wise to Xk (n_samples, k) features already scaled to [0, pi]."""
    batch, k = Xk.shape
    n = int(np.ceil(k / 3))
    pad = 3 * n - k
    if pad:
        Xk = np.concatenate([Xk, np.zeros((batch, pad))], axis=1)
    psi = _init_batched_psi(batch, n)
    for q in range(n):
        fx, fy, fz = Xk[:, 3 * q], Xk[:, 3 * q + 1], Xk[:, 3 * q + 2]
        psi = _apply_1q_batched(psi, _batched_rot("RX", fx), q)
        psi = _apply_1q_batched(psi, _batched_rot("RY", fy), q)
        psi = _apply_1q_batched(psi, _batched_rot("RZ", fz), q)
    for q in range(n - 1):
        psi = _apply_cnot_batched(psi, q, q + 1)
    rng = np.random.default_rng(1234)
    phis = rng.uniform(0, 2 * np.pi, size=(n, 2))
    for q in range(n):
        for kind, col in (("RY", 0), ("RZ", 1)):
            theta = np.full(batch, phis[q, col])
            psi = _apply_1q_batched(psi, _batched_rot(kind, theta), q)
    return _batched_z_expectations(psi)


def shot_noisy_expectation(exact_exp, n_shots, rng):
    """Simulate finite-shot Z-measurement noise the way a real backend would
    produce it: draw the |1> count from Binomial(n_shots, p1) and rebuild the
    sample-mean expectation value. This reproduces the *statistics* of
    repeating the circuit n_shots times without literally looping n_shots
    times (numpy.random.binomial is exact for this purpose)."""
    p1 = (1 - exact_exp) / 2
    p1 = np.clip(p1, 0.0, 1.0)
    counts1 = rng.binomial(n_shots, p1)
    p1_hat = counts1 / n_shots
    return 1 - 2 * p1_hat
