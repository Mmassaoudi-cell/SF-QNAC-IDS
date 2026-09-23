"""
Core SF-QNAC-IDS mechanism, and every piece of the 7-variant ablation ladder,
implemented as ONE parameterised trainer so the ladder is a set of flag
toggles on a shared environment/update loop, not seven separate re-writes.

  use_quantum_actor    False -> classical MLP actor   | True -> VQC actor
  natural_gradient      False -> ordinary REINFORCE     | True -> diagonal-Fisher natural gradient (quantum params only)
  adaptive_shots         False -> fixed SHOTS_FIXED shots | True -> entropy/margin/phase-driven shot budget
  cost_aware_reward     False -> task-only reward        | True -> + quantum-shot-cost + depth-cost terms

Ladder mapping (see ABLATION_CONFIGS at bottom):
  4 classical-actor-critic : use_quantum_actor=False
  5 VQC actor-critic (ordinary grad) : quantum=True,  natural=False, adaptive=False, cost_aware=False
  6 QNAC (fixed shots)               : quantum=True,  natural=True,  adaptive=False, cost_aware=False
  7 SF-QNAC-IDS (full)               : quantum=True,  natural=True,  adaptive=True,  cost_aware=True

Rungs 1-3 (XGBoost / SHO-XGBoost / QRL-SHO-XGBoost reproduction) live in
baseline_repro.py, which is structurally different by design (they are not
ablations of this trainer -- rung 3 specifically reconstructs the reference
paper's own mechanism, per REFERENCE_PAPER_AUDIT.md).
"""
import time
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import xgboost as xgb

from quantum_sim import Circuit, GATE_FN

RNG_GLOBAL_SEED = 42

# ---------------------------------------------------------------------------
# Feature groups (contiguous chunks of the column list -- dataset-agnostic,
# documented as a simplification: not semantically curated per dataset).
# ---------------------------------------------------------------------------
def build_feature_groups(feature_names, n_groups=8):
    idx = np.arange(len(feature_names))
    chunks = np.array_split(idx, min(n_groups, len(feature_names)))
    return [list(c) for c in chunks if len(c) > 0]


# ---------------------------------------------------------------------------
# Multi-fidelity XGBoost evaluation
# ---------------------------------------------------------------------------
EVAL_COUNTER = {"low_fidelity": 0, "full_fidelity": 0}


def _fit_eval(Xtr, ytr, Xval, yval, hp, fidelity="low"):
    if fidelity == "low":
        EVAL_COUNTER["low_fidelity"] += 1
        n_sub = min(len(ytr), 1200)
        rng = np.random.default_rng(0)
        sel = rng.choice(len(ytr), n_sub, replace=False)
        Xtr_use, ytr_use = Xtr[sel], ytr[sel]
        n_est = min(hp["n_estimators"], 60)
    else:
        EVAL_COUNTER["full_fidelity"] += 1
        Xtr_use, ytr_use = Xtr, ytr
        n_est = hp["n_estimators"]
    clf = xgb.XGBClassifier(
        n_estimators=int(n_est), max_depth=int(hp["max_depth"]),
        learning_rate=hp["learning_rate"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        reg_alpha=hp["reg_alpha"], reg_lambda=hp["reg_lambda"],
        eval_metric="logloss", n_jobs=4, verbosity=0, random_state=RNG_GLOBAL_SEED,
    )
    clf.fit(Xtr_use, ytr_use)
    proba = clf.predict_proba(Xval)[:, 1]
    pred = (proba >= 0.5).astype(int)
    f1 = f1_score(yval, pred, average="macro", zero_division=0)
    try:
        pr_auc = average_precision_score(yval, proba)
    except ValueError:
        pr_auc = 0.5
    minority = 1 if yval.mean() > 0.5 else 0
    rec_min = ((pred == minority) & (yval == minority)).sum() / max((yval == minority).sum(), 1)
    fpr = ((pred == 1) & (yval == 0)).sum() / max((yval == 0).sum(), 1)
    return {"f1_macro": f1, "pr_auc": pr_auc, "minority_recall": rec_min, "fpr": fpr}, clf


DEFAULT_HP = dict(n_estimators=150, max_depth=5, learning_rate=0.1, min_child_weight=1,
                   gamma=0.0, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0)
HP_BOUNDS = dict(n_estimators=(50, 400), max_depth=(2, 10), learning_rate=(0.01, 0.3),
                  min_child_weight=(1, 10), gamma=(0.0, 5.0), subsample=(0.5, 1.0),
                  colsample_bytree=(0.5, 1.0), reg_alpha=(0.0, 5.0), reg_lambda=(0.1, 5.0))
HP_STEP = dict(n_estimators=25, max_depth=1, learning_rate=0.02, min_child_weight=1,
               gamma=0.25, subsample=0.05, colsample_bytree=0.05, reg_alpha=0.25, reg_lambda=0.25)
TUNED_HPS = ["max_depth", "learning_rate", "n_estimators", "min_child_weight", "gamma"]


class IDSConfigEnv:
    """Optimization-state RL environment: actions toggle feature groups or
    nudge a tuned XGBoost hyperparameter. State = compact validation-metric
    + config vector, NOT a per-sample quantum embedding."""

    def __init__(self, X_train, y_train, X_val, y_val, feature_names, n_groups=8, seed=0):
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.groups = build_feature_groups(feature_names, n_groups)
        self.n_groups = len(self.groups)
        self.rng = np.random.default_rng(seed)
        self.n_actions = self.n_groups + 2 * len(TUNED_HPS) + 1  # toggle-group | hp+/hp- | no-op
        self.reset()

    def reset(self):
        self.active = np.zeros(self.n_groups, dtype=bool)
        init_on = self.rng.choice(self.n_groups, size=max(2, self.n_groups // 2), replace=False)
        self.active[init_on] = True
        self.hp = dict(DEFAULT_HP)
        self.last_metrics = {"f1_macro": 0.0, "pr_auc": 0.5, "minority_recall": 0.0, "fpr": 1.0}
        self.step_count = 0
        return self.state()

    def _mask(self):
        mask = np.zeros(self.X_train.shape[1], dtype=bool)
        for gi, active in enumerate(self.active):
            if active:
                mask[self.groups[gi]] = True
        if not mask.any():
            mask[self.groups[0]] = True
        return mask

    def state(self):
        frac_selected = self.active.mean()
        hp_norm = [(self.hp[k] - HP_BOUNDS[k][0]) / (HP_BOUNDS[k][1] - HP_BOUNDS[k][0]) for k in TUNED_HPS]
        m = self.last_metrics
        s = np.array([
            m["f1_macro"], m["pr_auc"], m["minority_recall"], m["fpr"],
            frac_selected, *hp_norm, self.step_count / 60.0,
        ], dtype=float)
        return np.clip(s, -3, 3)

    def apply_action(self, a):
        if a < self.n_groups:
            self.active[a] = not self.active[a]
        elif a < self.n_groups + 2 * len(TUNED_HPS):
            j = a - self.n_groups
            key = TUNED_HPS[j // 2]
            sign = 1 if j % 2 == 0 else -1
            lo, hi = HP_BOUNDS[key]
            self.hp[key] = float(np.clip(self.hp[key] + sign * HP_STEP[key], lo, hi))
        # else: no-op

    def step(self, a, fidelity="low"):
        self.apply_action(a)
        mask = self._mask()
        metrics, _ = _fit_eval(self.X_train[:, mask], self.y_train, self.X_val[:, mask], self.y_val,
                                self.hp, fidelity=fidelity)
        prev = self.last_metrics
        d_f1 = metrics["f1_macro"] - prev["f1_macro"]
        self.last_metrics = metrics
        self.step_count += 1
        return self.state(), metrics, d_f1, mask


# ---------------------------------------------------------------------------
# Cost-aware reward (spec Sec. 11)
# ---------------------------------------------------------------------------
REWARD_WEIGHTS = dict(l1=1.0, l2=0.25, l3=0.25, l4=0.20, l5=0.05, l6=0.05, l7=0.05, l8=0.02)


def compute_reward(d_f1, metrics, prev_metrics, frac_selected, latency_norm, shots_used,
                    shots_ref, depth, depth_ref, cost_aware):
    r = (REWARD_WEIGHTS["l1"] * d_f1
         + REWARD_WEIGHTS["l2"] * (metrics["pr_auc"] - prev_metrics["pr_auc"])
         + REWARD_WEIGHTS["l3"] * (metrics["minority_recall"] - prev_metrics["minority_recall"])
         - REWARD_WEIGHTS["l4"] * metrics["fpr"]
         - REWARD_WEIGHTS["l5"] * frac_selected
         - REWARD_WEIGHTS["l6"] * latency_norm)
    if cost_aware:
        r -= REWARD_WEIGHTS["l7"] * (shots_used / max(shots_ref, 1))
        r -= REWARD_WEIGHTS["l8"] * (depth / max(depth_ref, 1))
    return float(r)


# ---------------------------------------------------------------------------
# Tiny numpy MLP (shared building block for classical actor + critic)
# ---------------------------------------------------------------------------
class TinyMLP:
    def __init__(self, in_dim, out_dim, hidden=16, seed=0, lr=0.02):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.3, (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.3, (hidden, out_dim))
        self.b2 = np.zeros(out_dim)
        self.lr = lr

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h1 = np.tanh(self.z1)
        self.out = self.h1 @ self.W2 + self.b2
        return self.out

    def backward(self, dout):
        dW2 = np.outer(self.h1, dout)
        db2 = dout
        dh1 = dout @ self.W2.T
        dz1 = dh1 * (1 - self.h1 ** 2)
        dW1 = np.outer(self.x, dz1)
        db1 = dz1
        self.W2 += self.lr * dW2; self.b2 += self.lr * db2
        self.W1 += self.lr * dW1; self.b1 += self.lr * db1


class Critic:
    """Classical critic V_w(s), always classical per spec Sec. 8."""
    def __init__(self, state_dim, seed=0):
        self.net = TinyMLP(state_dim, 1, hidden=16, seed=seed, lr=0.05)

    def value(self, s):
        return float(self.net.forward(s)[0])

    def update(self, s, td_error):
        self.net.forward(s)
        self.net.backward(np.array([td_error]))  # gradient ascent on -MSE == td_error * d(out)/dw


class ClassicalActor:
    """Baseline (non-quantum) actor for the ablation ladder's rung 4."""
    def __init__(self, state_dim, n_actions, seed=0):
        self.net = TinyMLP(state_dim, n_actions, hidden=16, seed=seed, lr=0.05)
        self.n_actions = n_actions

    def policy(self, s):
        logits = self.net.forward(s)
        logits = logits - logits.max()
        p = np.exp(logits); p /= p.sum()
        return p

    def update(self, s, a, advantage):
        p = self.policy(s)
        dlogits = -p.copy()
        dlogits[a] += 1.0  # d log pi(a|s) / d logits
        self.net.backward(advantage * dlogits)
        return p

    def circuit_stats(self):
        return {"qubits": 0, "depth": 0, "params": self.net.W1.size + self.net.W2.size}


# ---------------------------------------------------------------------------
# Quantum actor: data re-uploading VQC + classical linear readout
# ---------------------------------------------------------------------------
class QuantumActor:
    def __init__(self, state_dim, n_actions, n_qubits=4, layers=2, seed=0):
        self.n_qubits = n_qubits
        self.layers = layers
        self.n_actions = n_actions
        rng = np.random.default_rng(seed)
        # TRAIN-fitted linear compression of the RL state down to n_qubits dims (stand-in for a
        # PCA fit on collected TRAIN states; a fixed random projection is used here and stated
        # explicitly as the "TRAIN-only normalization" simplification for this phase).
        self.proj = rng.normal(0, 1.0 / np.sqrt(state_dim), (state_dim, n_qubits))
        # trainable data re-uploading scale/bias, per qubit per layer (2 params each)
        self.reupload_w = rng.normal(0, 0.5, (layers, n_qubits))
        self.reupload_b = rng.normal(0, 0.1, (layers, n_qubits))
        # trainable variational rotation angles RY, RZ per qubit per layer
        self.var_ry = rng.uniform(0, 2 * np.pi, (layers, n_qubits))
        self.var_rz = rng.uniform(0, 2 * np.pi, (layers, n_qubits))
        # classical linear readout: n_qubits expectation values -> n_actions logits
        self.readout_W = rng.normal(0, 0.3, (n_qubits, n_actions))
        self.readout_b = np.zeros(n_actions)
        self.lr_quantum = 0.08
        self.lr_readout = 0.05
        self.fisher_diag = {"reupload_w": np.ones_like(self.reupload_w) * 1e-2,
                             "reupload_b": np.ones_like(self.reupload_b) * 1e-2,
                             "var_ry": np.ones_like(self.var_ry) * 1e-2,
                             "var_rz": np.ones_like(self.var_rz) * 1e-2}
        self.fisher_decay = 0.95
        self.damping = 1e-2

    def _build_circuit(self, s_proj, reupload_w, reupload_b, var_ry, var_rz):
        circ = Circuit(self.n_qubits)
        for l in range(self.layers):
            for q in range(self.n_qubits):
                theta = reupload_w[l, q] * s_proj[q] + reupload_b[l, q]
                circ.add("RY", [q], float(theta))
            for q in range(self.n_qubits - 1):
                circ.add("CNOT", [q, q + 1])
            for q in range(self.n_qubits):
                circ.add("RY", [q], float(var_ry[l, q]))
                circ.add("RZ", [q], float(var_rz[l, q]))
        return circ

    def expectations(self, s, params=None):
        s_proj = np.tanh(s @ self.proj)
        p = params or dict(reupload_w=self.reupload_w, reupload_b=self.reupload_b,
                            var_ry=self.var_ry, var_rz=self.var_rz)
        circ = self._build_circuit(s_proj, p["reupload_w"], p["reupload_b"], p["var_ry"], p["var_rz"])
        return circ.z_expectations(), s_proj

    def policy(self, s, shots=None, shot_rng=None):
        z, s_proj = self.expectations(s)
        if shots is not None:
            from quantum_sim import shot_noisy_expectation
            z = shot_noisy_expectation(z, shots, shot_rng or np.random.default_rng())
        logits = z @ self.readout_W + self.readout_b
        logits = logits - logits.max()
        p = np.exp(logits); p /= p.sum()
        self._last = {"z": z, "s_proj": s_proj, "p": p}
        return p, z

    def build_circuit_for_state(self, s):
        """Exposes the actual Circuit object for a given state, for Phase-2
        noise-robustness evaluation (noise_sim.py operates on Circuit gate
        lists, not on already-collapsed expectation values)."""
        s_proj = np.tanh(s @ self.proj)
        return self._build_circuit(s_proj, self.reupload_w, self.reupload_b, self.var_ry, self.var_rz)

    def policy_from_z(self, z):
        """Classical readout only, given an externally-supplied (possibly
        noisy) expectation vector z -- lets a caller compare policies under
        different noise conditions through the identical trained readout."""
        logits = z @ self.readout_W + self.readout_b
        logits = logits - logits.max()
        p = np.exp(logits); p /= p.sum()
        return p

    def circuit_stats(self):
        circ = self._build_circuit(np.zeros(self.n_qubits), self.reupload_w, self.reupload_b,
                                    self.var_ry, self.var_rz)
        return {"qubits": self.n_qubits, "depth": circ.depth(),
                "gates_1q": circ.one_qubit_gate_count(), "gates_2q": circ.two_qubit_gate_count(),
                "params": self.reupload_w.size + self.reupload_b.size + self.var_ry.size + self.var_rz.size}

    def _param_shift_grad_z(self, s, param_name, l, q):
        """Exact parameter-shift gradient of every z_k w.r.t. one circuit parameter."""
        shift = np.pi / 2
        p_plus = dict(reupload_w=self.reupload_w.copy(), reupload_b=self.reupload_b.copy(),
                       var_ry=self.var_ry.copy(), var_rz=self.var_rz.copy())
        p_minus = dict(reupload_w=self.reupload_w.copy(), reupload_b=self.reupload_b.copy(),
                        var_ry=self.var_ry.copy(), var_rz=self.var_rz.copy())
        p_plus[param_name][l, q] += shift
        p_minus[param_name][l, q] -= shift
        z_plus, _ = self.expectations(s, p_plus)
        z_minus, _ = self.expectations(s, p_minus)
        return (z_plus - z_minus) / 2.0

    def update(self, s, a, advantage, natural_gradient=True, param_shift_budget="full"):
        """One-step actor-critic policy-gradient update.
        param_shift_budget: 'full' evaluates parameter-shift for every quantum
        parameter (used here since our exact simulator makes this numerically
        cheap); a real-hardware deployment would need 2 circuit executions per
        parameter per shift -- the resulting circuit-evaluation count is
        tracked in EVAL_COUNTER-style bookkeeping by the caller."""
        p, z = self.policy(s)
        dlogp_dz = self.readout_W[:, a] - (p[np.newaxis, :] @ self.readout_W.T).flatten()
        # ^ d log pi(a|s) / dz_k = W[k,a] - sum_a' p(a') W[k,a']
        n_shift_evals = 0
        grads = {}
        for name, arr in [("reupload_w", self.reupload_w), ("reupload_b", self.reupload_b),
                           ("var_ry", self.var_ry), ("var_rz", self.var_rz)]:
            g = np.zeros_like(arr)
            for l in range(self.layers):
                for q in range(self.n_qubits):
                    dz = self._param_shift_grad_z(s, name, l, q)
                    n_shift_evals += 2
                    g[l, q] = dz @ dlogp_dz
            grads[name] = g
        for name in grads:
            fd = self.fisher_diag[name]
            fd *= self.fisher_decay
            fd += (1 - self.fisher_decay) * (grads[name] ** 2)
            step = grads[name] / (fd + self.damping) if natural_gradient else grads[name]
            arr = getattr(self, name)
            arr += self.lr_quantum * advantage * step
        # classical readout: ordinary gradient always
        dlogits = -p.copy(); dlogits[a] += 1.0
        self.readout_W += self.lr_readout * advantage * np.outer(z, dlogits)
        self.readout_b += self.lr_readout * advantage * dlogits
        return n_shift_evals


# ---------------------------------------------------------------------------
# Adaptive shot allocation
# ---------------------------------------------------------------------------
SHOTS_FIXED = 8192


def adaptive_shots(policy_probs, phase_frac):
    p_sorted = np.sort(policy_probs)[::-1]
    margin = p_sorted[0] - (p_sorted[1] if len(p_sorted) > 1 else 0.0)
    entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-12)) / np.log(len(policy_probs))
    if phase_frac > 0.9:
        return 4096          # final policy confirmation
    if margin > 0.5:
        return 128            # one action clearly dominates
    if entropy > 0.85:
        return 256            # broad, intentional exploration
    if phase_frac < 0.33:
        return 256            # early exploration
    if phase_frac < 0.66:
        return 512            # intermediate
    return 1024                # late training, uncertain


# ---------------------------------------------------------------------------
# Trainer (shared by rungs 4-7 of the ablation ladder)
# ---------------------------------------------------------------------------
def run_actor_critic(env, budget, use_quantum_actor, natural_gradient, adaptive_shots_flag,
                      cost_aware_reward, n_qubits=4, layers=2, seed=0,
                      conv_eps=1e-3, patience=5, shots_fixed_override=None):
    """shots_fixed_override: Phase-2 shot-sensitivity sweep hook -- when set,
    used as the fixed shot count instead of the module-level SHOTS_FIXED
    (8192) whenever adaptive_shots_flag is False. Passed explicitly rather
    than via global-state monkeypatching."""
    global EVAL_COUNTER
    EVAL_COUNTER = {"low_fidelity": 0, "full_fidelity": 0}
    rng = np.random.default_rng(seed)
    shot_rng = np.random.default_rng(seed + 500)
    s = env.state()
    critic = Critic(len(s), seed=seed)
    if use_quantum_actor:
        actor = QuantumActor(len(s), env.n_actions, n_qubits=n_qubits, layers=layers, seed=seed)
    else:
        actor = ClassicalActor(len(s), env.n_actions, seed=seed)

    total_shots = 0
    total_circuit_evals = 0
    best_f1 = -1.0
    best_mask, best_hp = None, None
    prev_metrics = dict(env.last_metrics)
    f1_prev = 0.0
    conv_episode = budget
    patience_ct = 0
    t0 = time.time()

    for ep in range(1, budget + 1):
        phase_frac = ep / budget
        if use_quantum_actor:
            fixed_shots = shots_fixed_override if shots_fixed_override is not None else SHOTS_FIXED
            n_shots = adaptive_shots(actor.policy(s)[0], phase_frac) if adaptive_shots_flag else fixed_shots
            p, z = actor.policy(s, shots=n_shots, shot_rng=shot_rng)
            total_shots += n_shots * n_qubits
            total_circuit_evals += 1
        else:
            p = actor.policy(s)
        a = rng.choice(len(p), p=p)
        fidelity = "full" if (phase_frac > 0.85 or p[a] > 0.6) else "low"
        s2, metrics, d_f1, mask = env.step(a, fidelity=fidelity)

        frac_sel = env.active.mean()
        depth = actor.circuit_stats().get("depth", 0) if use_quantum_actor else 0
        r = compute_reward(d_f1, metrics, prev_metrics, frac_sel, latency_norm=0.1,
                            shots_used=(n_shots if use_quantum_actor else 0), shots_ref=SHOTS_FIXED,
                            depth=depth, depth_ref=12, cost_aware=cost_aware_reward)

        v_s, v_s2 = critic.value(s), critic.value(s2)
        td_target = r + 0.95 * v_s2
        td_error = td_target - v_s
        critic.update(s, td_error)
        if use_quantum_actor:
            n_shift = actor.update(s, a, td_error, natural_gradient=natural_gradient)
            total_circuit_evals += n_shift
        else:
            actor.update(s, a, td_error)

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]; best_mask = mask.copy(); best_hp = dict(env.hp)
        if abs(metrics["f1_macro"] - f1_prev) < conv_eps:
            patience_ct += 1
            if patience_ct >= patience and conv_episode == budget:
                conv_episode = ep
        else:
            patience_ct = 0
        f1_prev = metrics["f1_macro"]
        prev_metrics = metrics
        s = s2

    wall_time = time.time() - t0
    return {
        "best_val_f1_macro": best_f1, "best_mask": best_mask, "best_hp": best_hp,
        "episodes_to_convergence": conv_episode, "budget": budget,
        "total_shots": int(total_shots), "total_circuit_evals": int(total_circuit_evals),
        "low_fidelity_evals": EVAL_COUNTER["low_fidelity"], "full_fidelity_evals": EVAL_COUNTER["full_fidelity"],
        "wall_time_sec": wall_time,
        "actor_stats": actor.circuit_stats(),
        "actor": actor, "final_state": s,  # for Phase-2 noise/sensitivity evaluation of the trained policy
    }


ABLATION_CONFIGS = {
    "4_classical_actor_critic": dict(use_quantum_actor=False, natural_gradient=False, adaptive_shots_flag=False, cost_aware_reward=False),
    "5_vqc_actor_critic_ordinary_grad": dict(use_quantum_actor=True, natural_gradient=False, adaptive_shots_flag=False, cost_aware_reward=False),
    "6_qnac_fixed_shots": dict(use_quantum_actor=True, natural_gradient=True, adaptive_shots_flag=False, cost_aware_reward=False),
    "7_sf_qnac_ids": dict(use_quantum_actor=True, natural_gradient=True, adaptive_shots_flag=True, cost_aware_reward=True),
}
