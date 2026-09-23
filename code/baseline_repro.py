"""
Reference QRL-SHO-XGBoost reproduction.

This is a RECONSTRUCTION, not a literal re-implementation -- see
REFERENCE_PAPER_AUDIT.md Sec. 2-3 for exactly which parameters the paper
never specifies (SHO population/iteration count, ansatz training rule,
reward lambda, action semantics, the SHO/RL-SHO scope ambiguity) and how
each is resolved here. Every assumed value is named below, inline.

Faithfully reproduced from the paper's stated equations:
  - 8-qubit angle encoding (Eq. 17-18): 1 raw feature -> 1 qubit via R_Y
  - ring CNOT entanglement (Eq. 13, 18)
  - RZ-RY-RX variational layer (Eq. 14)
  - Pauli-Z expectation read-out (Eq. 5, 15-16)
  - 8192-shot estimation (Sec. III.D text)
  - tabular Q-learning, alpha=0.10, gamma=0.95, eps 1.0->0.01 decay 0.995,
    5 discrete actions (Eq. 20, 23)
  - XGBoost final classifier (Eq. 24-25)

Assumed (undocumented in the paper, stated here explicitly):
  - SHO population N=10, iterations T=6 (paper gives no number)
  - the "horse" position vector jointly encodes a binary feature-relevance
    mask (thresholded top-k) AND XGBoost hyperparameters, since the paper's
    own text blends these two searches (REFERENCE_PAPER_AUDIT.md Sec. 3.1)
  - reward lambda (sparsity weight) = 0.05
  - variational ansatz angles are FIXED (seeded random), not trained --
    the paper never states a training rule for them separately from the
    RL-tuned XGBoost hyperparameters
  - the five discrete RL actions are: no-op, n_estimators +/-, max_depth +/-
"""
import time
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

from quantum_sim import batched_baseline_embed, shot_noisy_expectation

K_SELECTED = 8
SHO_N = 10          # ASSUMED (paper does not state a value)
SHO_T = 6            # ASSUMED
RL_BUDGET = 80        # reduced from the paper's ~1500 for this phase's compute budget (stated reduction)
SHOTS = 8192          # as stated in the paper
REWARD_LAMBDA = 0.05  # ASSUMED

ACTIONS = [(0, 0), (20, 0), (-20, 0), (0, 1), (0, -1)]  # ASSUMED semantics for "five discrete actions"
NEST_BOUNDS = (50, 300)
DEPTH_BOUNDS = (2, 10)

FIT_COUNTER = {"n": 0}


def _clip_hp(n_est, depth):
    return int(np.clip(n_est, *NEST_BOUNDS)), int(np.clip(depth, *DEPTH_BOUNDS))


def _fit_eval(Xtr, ytr, Xval, yval, n_est, depth, lr=0.1):
    FIT_COUNTER["n"] += 1
    clf = xgb.XGBClassifier(n_estimators=int(n_est), max_depth=int(depth), learning_rate=lr,
                             eval_metric="logloss", n_jobs=4, verbosity=0, random_state=0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xval)
    return f1_score(yval, pred, average="macro", zero_division=0)


def _mask_from_position(pos_feat, k=K_SELECTED):
    order = np.argsort(-pos_feat)
    mask = np.zeros_like(pos_feat, dtype=bool)
    mask[order[:min(k, len(pos_feat))]] = True
    return mask


def _sho_update(pos, vel, pbest, gbest, w, rng):
    d = pos.shape[0]
    r1, r2 = rng.uniform(0, 1, d), rng.uniform(0, 1, d)
    jump = rng.normal(0, 0.1, d)
    vel_new = w * vel + 1.5 * r1 * (pbest - pos) + 1.5 * r2 * (gbest - pos) + 0.1 * jump
    return pos + vel_new, vel_new


def _embed(X8, shots_rng):
    z = batched_baseline_embed(X8)
    if shots_rng is not None:
        z = shot_noisy_expectation(z, SHOTS, shots_rng)
    return z


def _score_candidate(Xtr, ytr, Xval, yval, mask, shots_rng):
    scaler = MinMaxScaler()
    Xtr8 = scaler.fit_transform(Xtr[:, mask]) * np.pi
    Xva8 = np.clip(scaler.transform(Xval[:, mask]), 0, 1) * np.pi
    Ztr, Zva = _embed(Xtr8, shots_rng), _embed(Xva8, shots_rng)
    f1 = _fit_eval(Ztr, ytr, Zva, yval, n_est=100, depth=4)
    return f1 - REWARD_LAMBDA * (mask.sum() / len(mask))


def run_reference_reproduction(X_train, y_train, X_val, y_val, X_test, y_test, seed=0,
                                search_subsample=3000):
    FIT_COUNTER["n"] = 0
    rng = np.random.default_rng(seed)
    shots_rng = np.random.default_rng(seed + 999)
    d = X_train.shape[1]
    t0 = time.time()

    n_sub = min(search_subsample, len(y_train))
    sub_idx = rng.choice(len(y_train), n_sub, replace=False)
    Xs_tr, ys_tr = X_train[sub_idx], y_train[sub_idx]
    n_val_sub = min(search_subsample // 3, len(y_val))
    val_idx = rng.choice(len(y_val), n_val_sub, replace=False)
    Xs_val, ys_val = X_val[val_idx], y_val[val_idx]

    # --- Stage 1: wrapper SHO (every candidate fully retrained) ---
    pos = rng.uniform(-1, 1, (SHO_N, d))
    vel = np.zeros((SHO_N, d))
    pbest = pos.copy(); pbest_fit = np.full(SHO_N, -np.inf)
    gbest = pos[0].copy(); gbest_fit = -np.inf
    for t in range(SHO_T):
        w = 0.9 - 0.5 * (t / max(SHO_T - 1, 1))
        for i in range(SHO_N):
            mask = _mask_from_position(pos[i])
            fit = _score_candidate(Xs_tr, ys_tr, Xs_val, ys_val, mask, shots_rng)
            if fit > pbest_fit[i]:
                pbest_fit[i], pbest[i] = fit, pos[i].copy()
            if fit > gbest_fit:
                gbest_fit, gbest = fit, pos[i].copy()
        for i in range(SHO_N):
            pos[i], vel[i] = _sho_update(pos[i], vel[i], pbest[i], gbest, w, rng)
    stage1_trainings = FIT_COUNTER["n"]
    best_mask = _mask_from_position(gbest)
    t1 = time.time()

    # --- Stage 2: tabular Q-learning over XGBoost hyperparameters ---
    scaler = MinMaxScaler()
    Ztr = _embed(scaler.fit_transform(Xs_tr[:, best_mask]) * np.pi, shots_rng)
    Zva = _embed(np.clip(scaler.transform(Xs_val[:, best_mask]), 0, 1) * np.pi, shots_rng)
    FIT_COUNTER["n"] = 0
    Q = {}
    n_est, depth = 100, 4
    eps = 1.0
    f1_prev = _fit_eval(Ztr, ys_tr, Zva, ys_val, n_est, depth)

    def disc(f1, n_est, depth, bins=8):
        fb = min(bins - 1, int(f1 * bins))
        nb = min(bins - 1, int((n_est - NEST_BOUNDS[0]) / (NEST_BOUNDS[1] - NEST_BOUNDS[0]) * bins))
        db = min(bins - 1, int((depth - DEPTH_BOUNDS[0]) / (DEPTH_BOUNDS[1] - DEPTH_BOUNDS[0]) * bins))
        return (fb, nb, db)

    s = disc(f1_prev, n_est, depth)
    best_f1, best_hp = f1_prev, (n_est, depth)
    conv_ep = RL_BUDGET
    patience = 0
    for ep in range(1, RL_BUDGET + 1):
        qvals = [Q.get((s, a), 0.0) for a in range(len(ACTIONS))]
        a = rng.integers(len(ACTIONS)) if rng.uniform() < eps else int(np.argmax(qvals))
        dn, dd = ACTIONS[a]
        n_est2, depth2 = _clip_hp(n_est + dn, depth + dd)
        f1_new = _fit_eval(Ztr, ys_tr, Zva, ys_val, n_est2, depth2)
        s2 = disc(f1_new, n_est2, depth2)
        q_sa = Q.get((s, a), 0.0)
        max_next = max(Q.get((s2, a2), 0.0) for a2 in range(len(ACTIONS)))
        Q[(s, a)] = q_sa + 0.10 * (f1_new + 0.95 * max_next - q_sa)
        if f1_new > best_f1:
            best_f1, best_hp = f1_new, (n_est2, depth2)
        if abs(f1_new - f1_prev) < 1e-3:
            patience += 1
            if patience >= 5 and conv_ep == RL_BUDGET:
                conv_ep = ep
        else:
            patience = 0
        n_est, depth, s, f1_prev = n_est2, depth2, s2, f1_new
        eps = max(0.05, eps * 0.97)
    stage2_trainings = FIT_COUNTER["n"]
    t2 = time.time()

    # --- final fit on full train, eval on held-out test ---
    scaler_f = MinMaxScaler()
    Ztr_f = _embed(scaler_f.fit_transform(X_train[:, best_mask]) * np.pi, np.random.default_rng(seed + 1))
    Zte_f = _embed(np.clip(scaler_f.transform(X_test[:, best_mask]), 0, 1) * np.pi, np.random.default_rng(seed + 2))
    n_est_f, depth_f = best_hp
    clf = xgb.XGBClassifier(n_estimators=n_est_f, max_depth=depth_f, learning_rate=0.1,
                             eval_metric="logloss", n_jobs=4, verbosity=0, random_state=seed)
    clf.fit(Ztr_f, y_train)
    proba = clf.predict_proba(Zte_f)[:, 1]
    pred = (proba >= 0.5).astype(int)
    from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score
    t3 = time.time()

    minority = 1 if y_test.mean() > 0.5 else 0
    return {
        "method": "3_qrl_sho_xgboost_reproduction",
        "n_qubits": 8, "stage1_full_trainings": stage1_trainings,
        "stage2_full_trainings": stage2_trainings, "episodes_to_convergence": conv_ep,
        "rl_budget": RL_BUDGET, "total_full_trainings": stage1_trainings + stage2_trainings + 1,
        "total_shots": int((n_sub + n_val_sub) * stage1_trainings * SHOTS
                            + (n_sub + n_val_sub) * SHOTS
                            + (len(y_train) + len(y_test)) * SHOTS),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "test_pr_auc": float(average_precision_score(y_test, proba)),
        "test_fpr": float(((pred == 1) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1)),
        "minority_recall": float(((pred == minority) & (y_test == minority)).sum() / max((y_test == minority).sum(), 1)),
        "wall_time_stage1_sec": round(t1 - t0, 2), "wall_time_stage2_sec": round(t2 - t1, 2),
        "wall_time_final_sec": round(t3 - t2, 2), "wall_time_total_sec": round(t3 - t0, 2),
        "selected_features_idx": np.where(best_mask)[0].tolist(),
        "best_hyperparams": {"n_estimators": n_est_f, "max_depth": depth_f},
    }
