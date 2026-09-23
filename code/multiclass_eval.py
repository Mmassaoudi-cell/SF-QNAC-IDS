"""
Phase 2: multiclass evaluation. The Phase-1 data pipeline binarizes labels
at load time (by design, for the binary-detection headline comparison), so
this module adds separate, label-preserving loaders for the same three
datasets and evaluates a REAL multiclass XGBoost classifier trained on the
raw attack-type labels.

Scope decision, stated explicitly: rather than re-running the full RL search
under a multiclass reward (a materially different optimization problem this
phase does not attempt), we reuse the feature groups and hyperparameters
SF-QNAC-IDS already found via the binary-label search (Phase 1,
results/raw_results.jsonl) and retrain the final classifier with a
multi:softprob objective on the real multiclass labels. This measures
whether a binary-optimized feature/hyperparameter choice holds up under
multiclass evaluation -- a real, meaningful question -- without claiming to
have performed multiclass-aware RL optimization, which is not what was run.

Rare-class handling, stated explicitly: classes are used exactly as they
occur in the source data (no synthetic balancing). Extremely rare classes
(single-digit-to-low-double-digit total occurrences, e.g. CIC-IDS-2017's
"Heartbleed", 11 total rows in the full source file) may end up with zero
or very few samples after working-subsample stratification; where this
happens it is reported as a real class-count limitation, not hidden by
merging rare classes into an "Other" bucket.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              precision_score, recall_score, confusion_matrix,
                              classification_report)
import xgboost as xgb

import data_pipeline as dp
from qnac_core import IDSConfigEnv, run_actor_critic, ABLATION_CONFIGS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
MC_WORKING_N = 15000


def _derive_sf_qnac_config(X_arr, y_multi, feat_names, benign_label, seed=42, search_n=3000):
    """SF-QNAC-IDS's ablation-ladder result rows (Phase 1) do not persist the
    feature mask for the actor-critic rungs (only best_hyperparams), and the
    binary and multiclass loaders here are built independently, so indices
    from a Phase-1 binary run cannot be safely reused. Instead this re-runs
    the identical SF-QNAC-IDS search (same env/actor/reward code) on a
    binary view of THIS multiclass data (benign vs. any attack) to get a
    feature mask + hyperparameters, then hands them to a multiclass
    classifier below -- self-consistent by construction, still a real
    search, not a re-used stale index list."""
    rng = np.random.default_rng(seed)
    y_bin = (y_multi != benign_label).astype(int)
    n_sub = min(search_n, len(y_bin))
    idx = rng.choice(len(y_bin), n_sub, replace=False)
    val_idx = rng.choice(len(y_bin), min(search_n // 3, len(y_bin)), replace=False)
    env = IDSConfigEnv(X_arr[idx], y_bin[idx], X_arr[val_idx], y_bin[val_idx],
                        feat_names, n_groups=8, seed=seed)
    res = run_actor_critic(env, budget=40, seed=seed, n_qubits=4, layers=2,
                            **ABLATION_CONFIGS["7_sf_qnac_ids"])
    mask = res["best_mask"] if res["best_mask"] is not None else np.ones(X_arr.shape[1], dtype=bool)
    hp = res["best_hp"] or dict(n_estimators=150, max_depth=5, learning_rate=0.1)
    return np.where(mask)[0].tolist(), hp


# --------------------------------------------------------------------------
# Label-preserving loaders (parallel to data_pipeline.py's binary ones)
# --------------------------------------------------------------------------
def load_unsw_multiclass():
    X, y_bin, y_multi, feat_names = dp.load_unsw_raw()
    X, y_multi, n_dropped = dp.deduplicate(X, y_multi)
    return X, y_multi, feat_names


def load_cic_multiclass():
    """Reads each attack-day file IN FULL (not through the binary pipeline's
    subsampled pool cache, which discards the original multiclass label at
    caching time) -- CIC-IDS-2017's individual day files are small enough
    (<=225MB) that a full read is feasible for this bounded Phase-2 check."""
    cache_path = CACHE_DIR / "cic2017_multiclass.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        y = df["__label__"].values
        X = df.drop(columns=["__label__"])
        return X, y, list(X.columns)

    parts_X, parts_y = [], []
    for day_key, fname in dp.CIC_FILES.items():
        d = pd.read_csv(dp.CIC_DIR / fname)
        d.columns = [c.strip() for c in d.columns]
        y = d["Label"].astype(str).str.strip().values
        Xd = d.drop(columns=["Label"]).replace([np.inf, -np.inf], np.nan)
        Xd = dp._encode_frame(Xd)
        parts_X.append(Xd); parts_y.append(y)
    X = pd.concat(parts_X, ignore_index=True)
    y = np.concatenate(parts_y)
    X, y, n_dropped = dp.deduplicate(X, y)

    out = X.copy(); out["__label__"] = y
    out.to_parquet(cache_path, index=False)
    return X, y, list(X.columns)


def load_ciciot23_multiclass():
    cache_path = CACHE_DIR / "ciciot23_multiclass.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        y = df["__label__"].values
        X = df.drop(columns=["__label__"])
        return X, y, list(X.columns)

    def label_fn(chunk):
        return chunk["label"].astype(str).values

    Xtr, ytr = dp.read_csv_stratified_by_chunks(
        dp.CICIOT_DIR / "train" / "train.csv", label_fn, MC_WORKING_N, dp.POOL_SEED,
        chunksize=300000, max_chunks=8)
    Xtr = Xtr.drop(columns=["label"]).replace([np.inf, -np.inf], np.nan)
    Xtr = dp._encode_frame(Xtr)
    X, y, n_dropped = dp.deduplicate(Xtr, ytr)

    out = X.copy(); out["__label__"] = y
    out.to_parquet(cache_path, index=False)
    return X, y, list(X.columns)


# --------------------------------------------------------------------------
def run_multiclass(dataset, X, y, feat_names, benign_label, seed=42, working_n=MC_WORKING_N):
    vals, counts = np.unique(y, return_counts=True)
    class_counts_full = dict(zip(vals.tolist(), counts.tolist()))

    X_arr = X.values.astype(float)
    Xs, ys = dp.stratified_subsample(X, y, min(working_n, len(y)), seed)
    Xs_arr = Xs.values.astype(float)

    mask_idx, hp = _derive_sf_qnac_config(Xs_arr, ys, feat_names, benign_label, seed=seed)

    vals2, counts2 = np.unique(ys, return_counts=True)
    keep_classes = vals2[counts2 >= 3]  # need >=3 to survive a 70/15/15 stratified split
    keep_mask = np.isin(ys, keep_classes)
    dropped_classes = sorted(set(vals2.tolist()) - set(keep_classes.tolist()))
    Xs_arr, ys = Xs_arr[keep_mask], ys[keep_mask]

    le_classes = sorted(set(ys.tolist()))
    class_to_idx = {c: i for i, c in enumerate(le_classes)}
    y_int = np.array([class_to_idx[c] for c in ys])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        Xs_arr, y_int, test_size=0.15, stratify=y_int, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, stratify=y_train_full, random_state=seed)

    clf = xgb.XGBClassifier(
        n_estimators=int(hp.get("n_estimators", 200)), max_depth=int(hp.get("max_depth", 6)),
        learning_rate=hp.get("learning_rate", 0.1), objective="multi:softprob",
        num_class=len(le_classes), eval_metric="mlogloss", n_jobs=4, verbosity=0,
        random_state=seed)
    clf.fit(X_train_full[:, mask_idx], y_train_full)
    pred = clf.predict(X_test[:, mask_idx])

    report = classification_report(y_test, pred, target_names=[str(c) for c in le_classes],
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=list(range(len(le_classes))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    return {
        "dataset": dataset, "n_classes_full_data": len(vals), "n_classes_evaluated": len(le_classes),
        "classes_dropped_insufficient_samples": dropped_classes,
        "class_counts_in_full_source_data": class_counts_full,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "class_names": [str(c) for c in le_classes],
        "n_train": len(y_train_full), "n_test": len(y_test),
    }


def main():
    results = {}
    print("=== UNSW-NB15 multiclass ===")
    X, y, feat = load_unsw_multiclass()
    r = run_multiclass("UNSW-NB15", X, y, feat, benign_label="Normal")
    results["UNSW-NB15"] = r
    print(f"classes: {r['n_classes_evaluated']}/{r['n_classes_full_data']} "
          f"(dropped: {r['classes_dropped_insufficient_samples']})")
    print(f"acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} macro_f1={r['macro_f1']:.4f}")

    print("\n=== CIC-IDS-2017 multiclass ===")
    X, y, feat = load_cic_multiclass()
    r = run_multiclass("CIC-IDS-2017", X, y, feat, benign_label="BENIGN")
    results["CIC-IDS-2017"] = r
    print(f"classes: {r['n_classes_evaluated']}/{r['n_classes_full_data']} "
          f"(dropped: {r['classes_dropped_insufficient_samples']})")
    print(f"acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} macro_f1={r['macro_f1']:.4f}")

    print("\n=== CICIoT23 multiclass ===")
    X, y, feat = load_ciciot23_multiclass()
    r = run_multiclass("CICIoT23", X, y, feat, benign_label="BenignTraffic")
    results["CICIoT23"] = r
    print(f"classes: {r['n_classes_evaluated']}/{r['n_classes_full_data']} "
          f"(dropped: {r['classes_dropped_insufficient_samples']})")
    print(f"acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} macro_f1={r['macro_f1']:.4f}")

    with open(RESULTS_DIR / "multiclass_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'multiclass_results.json'}")


if __name__ == "__main__":
    main()
