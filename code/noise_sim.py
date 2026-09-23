"""
NISQ-like noise model for the (small, <=6-qubit) circuits used throughout,
built on top of quantum_sim.Circuit's gate list.

Design choice, stated explicitly: depolarizing noise is simulated by
Monte-Carlo Pauli-error trajectories (after each single-qubit gate, with
probability p_1q, apply a uniformly-random Pauli {X,Y,Z}; after each CNOT,
independently apply the same single-qubit kick with probability p_2q to
each of the two qubits it touches), averaged over n_trajectories runs. This
is a standard, cheap "unraveling" of a depolarizing channel and is exact in
the n_trajectories -> infinity limit; it is NOT a full density-matrix
simulation, which would be more exact but far more expensive for the
sensitivity/robustness sweep this module exists to run cheaply. Readout
noise uses the standard symmetric bit-flip model, which has a closed-form
effect on a Z-expectation value: <Z>_noisy = (1-2p_ro) <Z>_ideal -- so it is
applied analytically (exact, no extra trajectories needed) rather than by
Monte Carlo, and readout-error "mitigation" is exact division by
(1-2p_ro), matching the standard calibration-matrix-inversion approach for
symmetric readout noise.
"""
import numpy as np

from quantum_sim import GATE_FN, rx, ry, rz, X, Z

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULIS = {"X": X, "Y": Y, "Z": Z}


def _apply_1q(psi, U, qubit, n):
    psi = np.tensordot(U, psi, axes=([1], [qubit]))
    return np.moveaxis(psi, 0, qubit)


def _apply_cnot(psi, control, target):
    psi = np.moveaxis(psi, [control, target], [0, 1])
    out = psi.copy()
    out[1, 0] = psi[1, 1]
    out[1, 1] = psi[1, 0]
    return np.moveaxis(out, [0, 1], [control, target])


def _random_pauli_kick(psi, qubit, n, rng):
    p = PAULIS[rng.choice(["X", "Y", "Z"])]
    return _apply_1q(psi, p, qubit, n)


def _run_one_trajectory(circuit, p_1q, p_2q, rng):
    n = circuit.n
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    psi = psi.reshape([2] * n)
    for name, qubits, theta in circuit.gates:
        if name in GATE_FN:
            (q,) = qubits
            psi = _apply_1q(psi, GATE_FN[name](theta), q, n)
            if rng.uniform() < p_1q:
                psi = _random_pauli_kick(psi, q, n, rng)
        elif name == "CNOT":
            c, t = qubits
            psi = _apply_cnot(psi, c, t)
            if rng.uniform() < p_2q:
                psi = _random_pauli_kick(psi, c, n, rng)
            if rng.uniform() < p_2q:
                psi = _random_pauli_kick(psi, t, n, rng)
        else:
            raise ValueError(name)
    return psi.reshape(-1)


def _z_expectations_from_state(psi, n):
    probs = np.abs(psi) ** 2
    idx = np.arange(2 ** n)
    exps = np.zeros(n)
    for q in range(n):
        bit = (idx >> (n - 1 - q)) & 1
        exps[q] = probs[bit == 0].sum() - probs[bit == 1].sum()
    return exps


def depolarizing_z_expectations(circuit, p_1q, p_2q, n_trajectories, rng):
    """Monte-Carlo depolarizing-noise Z-expectations (see module docstring)."""
    n = circuit.n
    acc = np.zeros(n)
    for _ in range(n_trajectories):
        psi = _run_one_trajectory(circuit, p_1q, p_2q, rng)
        acc += _z_expectations_from_state(psi, n)
    return acc / n_trajectories


def readout_noisy(z_ideal, p_readout):
    """Analytic symmetric bit-flip readout noise: <Z>_noisy = (1-2p)<Z>_ideal."""
    return (1 - 2 * p_readout) * np.asarray(z_ideal)


def readout_mitigated(z_noisy, p_readout):
    """Exact inversion of the symmetric bit-flip model (closed-form
    calibration-matrix inversion for this special case)."""
    denom = (1 - 2 * p_readout)
    if abs(denom) < 1e-9:
        return z_noisy
    return np.asarray(z_noisy) / denom


def zne_extrapolate(circuit, p_1q, p_2q, n_trajectories, rng, scales=(1, 3)):
    """Zero-noise extrapolation: evaluate at noise-scale factors (equivalent
    to gate folding) and linearly extrapolate back to zero noise. Only meant
    to be called at final-policy-evaluation time (expensive relative to a
    single noisy read-out), matching the design brief's guidance to reserve
    ZNE for final refinement/evaluation, not every RL update."""
    ys = []
    for lam in scales:
        z = depolarizing_z_expectations(circuit, p_1q * lam, p_2q * lam, n_trajectories, rng)
        ys.append(z)
    # linear fit per-qubit through the |scales| points, extrapolate to 0
    scales = np.array(scales, dtype=float)
    ys = np.array(ys)  # (len(scales), n_qubits)
    if len(scales) == 2:
        slope = (ys[1] - ys[0]) / (scales[1] - scales[0])
        intercept = ys[0] - slope * scales[0]
        return intercept
    # generic least-squares linear extrapolation for >2 points
    A = np.vstack([scales, np.ones_like(scales)]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return coef[1]  # intercept term


def combined_noise_z(circuit, condition, rng, p_1q=0.01, p_2q=0.03, p_readout=0.02,
                      n_trajectories=200, mitigation=None):
    """condition in {'ideal', 'depolarizing', 'readout', 'combined'}.
    mitigation in {None, 'readout', 'zne'}."""
    n = circuit.n
    if condition == "ideal":
        return circuit.z_expectations()
    if condition == "depolarizing":
        return depolarizing_z_expectations(circuit, p_1q, p_2q, n_trajectories, rng)
    if condition == "readout":
        z = circuit.z_expectations()
        zn = readout_noisy(z, p_readout)
        if mitigation == "readout":
            zn = readout_mitigated(zn, p_readout)
        return zn
    if condition == "combined":
        if mitigation == "zne":
            zdep = zne_extrapolate(circuit, p_1q, p_2q, n_trajectories, rng)
        else:
            zdep = depolarizing_z_expectations(circuit, p_1q, p_2q, n_trajectories, rng)
        zn = readout_noisy(zdep, p_readout)
        if mitigation == "readout":
            zn = readout_mitigated(zn, p_readout)
        return zn
    raise ValueError(condition)
