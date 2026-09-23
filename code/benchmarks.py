"""
6-method benchmark suite (scoped down from the full ~25-method spec for this
phase, per agreed focus): LogisticRegression, RandomForest, LightGBM,
XGBoost (default), Optuna-tuned XGBoost (optimization-control baseline),
and a quantum-kernel classifier (quantum/hybrid baseline).

The quantum-kernel classifier is deliberately a SEPARATE use of quantum
computation from SF-QNAC-IDS: here the quantum circuit embeds every sample
(a per-sample quantum feature map feeding a classical linear classifier,
i.e. the QRL-SHO-XGBoost-style usage), whereas SF-QNAC-IDS's quantum actor
only ever touches the compact RL optimization state. Keeping the two
separate is intentional (see Sec. 28 of the task spec: don't conflate
classifier explainability/usage with quantum-policy usage).
"""
import time
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                              average_precision_score, accuracy_score)

from quantum_sim import batched_squire_embed

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _metrics(y_test, pred, proba):
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


def _run(name, fit_fn, X_train, y_train, X_test, y_test, extra=None):
    t0 = time.time()
    proba, n_trainings = fit_fn(X_train, y_train, X_test)
    pred = (proba >= 0.5).astype(int)
    m = _metrics(y_test, pred, proba)
    m.update({"method": name, "wall_time_total_sec": round(time.time() - t0, 2),
              "total_full_trainings": n_trainings})
    if extra:
        m.update(extra)
    return m


def run_logreg(X_train, y_train, X_val, y_val, X_test, y_test, seed=0):
    def fit_fn(Xtr, ytr, Xte):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1], 1
    return _run("bench_logreg", fit_fn, X_train, y_train, X_test, y_test)


def run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, seed=0):
    def fit_fn(Xtr, ytr, Xte):
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      n_jobs=4, random_state=seed)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1], 1
    return _run("bench_random_forest", fit_fn, X_train, y_train, X_test, y_test)


def run_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, seed=0):
    def fit_fn(Xtr, ytr, Xte):
        clf = lgb.LGBMClassifier(n_estimators=200, class_weight="balanced",
                                  n_jobs=4, random_state=seed, verbosity=-1)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1], 1
    return _run("bench_lightgbm", fit_fn, X_train, y_train, X_test, y_test)


def run_xgboost_default(X_train, y_train, X_val, y_val, X_test, y_test, seed=0):
    def fit_fn(Xtr, ytr, Xte):
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 eval_metric="logloss", n_jobs=4, verbosity=0, random_state=seed)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1], 1
    return _run("1_xgboost_default", fit_fn, X_train, y_train, X_test, y_test)


def run_optuna_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, seed=0, n_trials=25):
    """Optimization-control baseline: does classical HPO alone match the
    quantum-RL search? Tuned on TRAIN/VAL only, exactly like SF-QNAC-IDS."""
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 400),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        )
        clf = xgb.XGBClassifier(**params, eval_metric="logloss", n_jobs=4, verbosity=0, random_state=seed)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        return f1_score(y_val, pred, average="macro", zero_division=0)

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    search_time = time.time() - t0

    best = study.best_params
    clf = xgb.XGBClassifier(**best, eval_metric="logloss", n_jobs=4, verbosity=0, random_state=seed)
    clf.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = _metrics(y_test, pred, proba)
    m.update({"method": "bench_optuna_xgboost", "wall_time_total_sec": round(time.time() - t0, 2),
              "wall_time_search_sec": round(search_time, 2),
              "total_full_trainings": n_trials + 1, "best_hyperparams": best})
    return m


def run_quantum_kernel_classifier(X_train, y_train, X_val, y_val, X_test, y_test, seed=0, k=8):
    """Quantum/hybrid baseline: mutual-info top-k raw features (NOT an RL
    search) -> fixed quantum feature map (dense tri-axis encoder, same
    circuit family as Squire-IDS/SF-QNAC-IDS's efficient encoder) ->
    classical LogisticRegression. A per-sample quantum embedding, unlike
    SF-QNAC-IDS's optimization-state-only quantum usage."""
    t0 = time.time()
    mi = mutual_info_classif(X_train, y_train, random_state=seed, n_neighbors=3)
    top_idx = np.argsort(-mi)[:k]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(X_train[:, top_idx]) * np.pi
    Xte = np.clip(scaler.transform(X_test[:, top_idx]), 0, 1) * np.pi
    Ztr = batched_squire_embed(Xtr)
    Zte = batched_squire_embed(Xte)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(Ztr, y_train)
    proba = clf.predict_proba(Zte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = _metrics(y_test, pred, proba)
    n_qubits = int(np.ceil(k / 3))
    m.update({"method": "bench_quantum_kernel_classifier", "wall_time_total_sec": round(time.time() - t0, 2),
              "total_full_trainings": 1, "n_qubits": n_qubits,
              "total_shots": 0, "note": "analytic (shot-free) quantum feature map, classical simulate-only deployment"})
    return m


BENCHMARK_RUNNERS = {
    "bench_logreg": run_logreg,
    "bench_random_forest": run_random_forest,
    "bench_lightgbm": run_lightgbm,
    "1_xgboost_default": run_xgboost_default,
    "bench_optuna_xgboost": run_optuna_xgboost,
    "bench_quantum_kernel_classifier": run_quantum_kernel_classifier,
}
