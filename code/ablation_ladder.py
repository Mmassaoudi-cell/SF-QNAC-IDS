"""
Unifies all 7 ablation-ladder rungs behind one call signature so the
orchestrator (run_all.py) can loop over them identically.

  1  XGBoost (default hyperparameters, all raw features)
  2  SHO-XGBoost (wrapper SHO feature search, classical -- no quantum, no RL)
  3  QRL-SHO-XGBoost reproduction (baseline_repro.py)
  4  Classical actor-critic sparse-XGBoost optimizer
  5  VQC actor-critic, ordinary gradient
  6  QNAC (quantum natural actor-critic), fixed shots
  7  SF-QNAC-IDS (full: QNAC + adaptive shots + cost-aware reward)

Rungs 4-7 share qnac_core.run_actor_critic with different flags -- this is
what makes the ladder a real ablation (isolating quantum actor / natural
gradient / shot adaptation / cost-aware reward one at a time) rather than
four independently-written methods that happen to be compared.
"""
import time
import numpy as np
import xgboost as xgb
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                              average_precision_score, accuracy_score)

import baseline_repro
from qnac_core import IDSConfigEnv, run_actor_critic, ABLATION_CONFIGS, _fit_eval as qnac_fit_eval


def _final_eval(X_train, y_train, X_test, y_test, mask, hp, seed=0):
    clf = xgb.XGBClassifier(
        n_estimators=int(hp.get("n_estimators", 150)), max_depth=int(hp.get("max_depth", 5)),
        learning_rate=hp.get("learning_rate", 0.1), min_child_weight=hp.get("min_child_weight", 1),
        gamma=hp.get("gamma", 0.0), subsample=hp.get("subsample", 0.9),
        colsample_bytree=hp.get("colsample_bytree", 0.9), reg_alpha=hp.get("reg_alpha", 0.0),
        reg_lambda=hp.get("reg_lambda", 1.0), eval_metric="logloss", n_jobs=4, verbosity=0,
        random_state=seed)
    clf.fit(X_train[:, mask], y_train)
    proba = clf.predict_proba(X_test[:, mask])[:, 1]
    pred = (proba >= 0.5).astype(int)
    minority = 1 if y_test.mean() > 0.5 else 0
    return {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "test_pr_auc": float(average_precision_score(y_test, proba)),
        "test_fpr": float(((pred == 1) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1)),
        "minority_recall": float(((pred == minority) & (y_test == minority)).sum() / max((y_test == minority).sum(), 1)),
    }


def run_rung1_xgboost(split, seed):
    t0 = time.time()
    mask = np.ones(split["X_train"].shape[1], dtype=bool)
    hp = dict(n_estimators=200, max_depth=6, learning_rate=0.1)
    m = _final_eval(split["X_train"], split["y_train"], split["X_test"], split["y_test"], mask, hp, seed)
    m.update({"method": "1_xgboost_default", "n_qubits": 0, "total_full_trainings": 1,
              "total_shots": 0, "episodes_to_convergence": 0,
              "wall_time_total_sec": round(time.time() - t0, 2)})
    return m


def run_rung2_sho_xgboost(split, seed, n_horses=10, n_iter=6, k=8, search_n=3000):
    """Wrapper SHO feature search (same swarm dynamics as the reference
    reproduction) but classical throughout -- no quantum embedding, no RL
    hyperparameter tuning. Isolates whether wrapper feature search alone
    (the paper's other major mechanism, independent of quantum/RL) helps."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    d = X_train.shape[1]
    n_sub = min(search_n, len(y_train))
    idx = rng.choice(len(y_train), n_sub, replace=False)
    Xs, ys = X_train[idx], y_train[idx]
    n_val_sub = min(search_n // 3, len(y_val))
    vidx = rng.choice(len(y_val), n_val_sub, replace=False)
    Xvs, yvs = X_val[vidx], y_val[vidx]

    fits = 0
    pos = rng.uniform(-1, 1, (n_horses, d))
    vel = np.zeros((n_horses, d))
    pbest = pos.copy(); pbest_fit = np.full(n_horses, -np.inf)
    gbest = pos[0].copy(); gbest_fit = -np.inf
    for t in range(n_iter):
        w = 0.9 - 0.5 * (t / max(n_iter - 1, 1))
        for i in range(n_horses):
            order = np.argsort(-pos[i]); mask = np.zeros(d, dtype=bool); mask[order[:k]] = True
            clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                     eval_metric="logloss", n_jobs=4, verbosity=0, random_state=seed)
            clf.fit(Xs[:, mask], ys)
            fits += 1
            fit = f1_score(yvs, clf.predict(Xvs[:, mask]), average="macro", zero_division=0)
            if fit > pbest_fit[i]:
                pbest_fit[i], pbest[i] = fit, pos[i].copy()
            if fit > gbest_fit:
                gbest_fit, gbest = fit, pos[i].copy()
        for i in range(n_horses):
            r1, r2 = rng.uniform(0, 1, d), rng.uniform(0, 1, d)
            jump = rng.normal(0, 0.1, d)
            vel[i] = w * vel[i] + 1.5 * r1 * (pbest[i] - pos[i]) + 1.5 * r2 * (gbest - pos[i]) + 0.1 * jump
            pos[i] = pos[i] + vel[i]
    order = np.argsort(-gbest); mask = np.zeros(d, dtype=bool); mask[order[:k]] = True
    hp = dict(n_estimators=200, max_depth=6, learning_rate=0.1)
    m = _final_eval(X_train, y_train, split["X_test"], split["y_test"], mask, hp, seed)
    m.update({"method": "2_sho_xgboost", "n_qubits": 0, "total_full_trainings": fits + 1,
              "total_shots": 0, "episodes_to_convergence": 0,
              "wall_time_total_sec": round(time.time() - t0, 2)})
    return m


def run_rung3_qrl_sho_xgboost_repro(split, seed):
    r = baseline_repro.run_reference_reproduction(
        split["X_train"], split["y_train"], split["X_val"], split["y_val"],
        split["X_test"], split["y_test"], seed=seed)
    r["episodes_to_convergence"] = r.pop("episodes_to_convergence")
    return r


def run_actor_critic_rung(name, split, seed, budget=60, n_groups=8, n_qubits=4, layers=2):
    cfg = ABLATION_CONFIGS[name]
    t0 = time.time()
    env = IDSConfigEnv(split["X_train"], split["y_train"], split["X_val"], split["y_val"],
                        split["feature_names"], n_groups=n_groups, seed=seed)
    res = run_actor_critic(env, budget=budget, seed=seed, n_qubits=n_qubits, layers=layers, **cfg)
    hp = res["best_hp"] if res["best_hp"] is not None else dict(n_estimators=150, max_depth=5,
                                                                  learning_rate=0.1, min_child_weight=1,
                                                                  gamma=0.0, subsample=0.9,
                                                                  colsample_bytree=0.9, reg_alpha=0.0,
                                                                  reg_lambda=1.0)
    mask = res["best_mask"] if res["best_mask"] is not None else np.ones(split["X_train"].shape[1], dtype=bool)
    m = _final_eval(split["X_train"], split["y_train"], split["X_test"], split["y_test"], mask, hp, seed)
    m.update({
        "method": name, "n_qubits": res["actor_stats"].get("qubits", 0),
        "circuit_depth": res["actor_stats"].get("depth", 0),
        "total_full_trainings": res["low_fidelity_evals"] + res["full_fidelity_evals"] + 1,
        "low_fidelity_evals": res["low_fidelity_evals"], "full_fidelity_evals": res["full_fidelity_evals"],
        "total_shots": res["total_shots"], "total_circuit_evals": res["total_circuit_evals"],
        "episodes_to_convergence": res["episodes_to_convergence"], "rl_budget": budget,
        "wall_time_total_sec": round(time.time() - t0, 2),
        "best_val_f1_macro": res["best_val_f1_macro"], "best_hyperparams": hp,
    })
    return m


RUNG_FUNCS = {
    "1_xgboost_default": lambda split, seed: run_rung1_xgboost(split, seed),
    "2_sho_xgboost": lambda split, seed: run_rung2_sho_xgboost(split, seed),
    "3_qrl_sho_xgboost_reproduction": lambda split, seed: run_rung3_qrl_sho_xgboost_repro(split, seed),
    "4_classical_actor_critic": lambda split, seed: run_actor_critic_rung("4_classical_actor_critic", split, seed),
    "5_vqc_actor_critic_ordinary_grad": lambda split, seed: run_actor_critic_rung("5_vqc_actor_critic_ordinary_grad", split, seed),
    "6_qnac_fixed_shots": lambda split, seed: run_actor_critic_rung("6_qnac_fixed_shots", split, seed),
    "7_sf_qnac_ids": lambda split, seed: run_actor_critic_rung("7_sf_qnac_ids", split, seed),
}
