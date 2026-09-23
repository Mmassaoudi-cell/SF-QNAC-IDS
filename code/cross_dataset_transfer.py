"""
Phase 2: cross-dataset transfer evaluation.

Feature reconciliation (done explicitly, not assumed): UNSW-NB15,
CIC-IDS-2017, and CICIoT23 have three unrelated feature schemas (42, 78, and
46 raw columns respectively, essentially no shared column names), so direct
per-feature transfer is not defensible. Instead we identify three raw
columns per dataset that are genuinely the same underlying network-flow
quantity under different names/units, and use ONLY those as the common
transfer representation:

  common feature   | UNSW-NB15         | CIC-IDS-2017                                          | CICIoT23
  duration          | dur                | Flow Duration                                          | flow_duration
  rate (pkts/sec)   | rate               | Flow Packets/s                                         | Rate
  total_bytes       | sbytes + dbytes    | Total Length of Fwd Packets + Total Length of Bwd Pkts | Tot size

Units/scales differ across datasets (e.g. CIC-IDS-2017's Flow Duration is
microseconds vs. UNSW's dur in seconds); each feature is independently
min-max-normalized PER DATASET (fit on that dataset's own train split) before
transfer, which is the best defensible reconciliation without assuming a
precise unit conversion this study did not verify against each dataset's raw
documentation. This is a stated simplification (Sec. "Limitations").

Reports, per (source, target) ordered pair, kept clearly separate:
  - in-domain: trained AND tested on the source dataset (upper bound)
  - zero-shot: trained on source, evaluated directly on target's test split,
    no target labels used at all
  - few-shot: zero-shot model additionally fine-tuned (continued training)
    on a small (n=300) labeled sample from the target's train split, then
    evaluated on the same target test split
  - performance drop = in-domain - zero-shot (on the SOURCE's own metric,
    i.e. how much the source model's own quality drops when asked to work
    on an unfamiliar domain's already-known-quantity feature space)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              average_precision_score, roc_auc_score, brier_score_loss)
import xgboost as xgb

import data_pipeline as dp
import multiclass_eval as mc

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FEW_SHOT_N = 300
SEED = 42

COMMON_COLS = {
    "UNSW-NB15": {"duration": ["dur"], "rate": ["rate"], "total_bytes": ["sbytes", "dbytes"]},
    "CIC-IDS-2017": {"duration": ["Flow Duration"], "rate": ["Flow Packets/s"],
                      "total_bytes": ["Total Length of Fwd Packets", "Total Length of Bwd Packets"]},
    "CICIoT23": {"duration": ["flow_duration"], "rate": ["Rate"], "total_bytes": ["Tot size"]},
}
BENIGN_LABEL = {"UNSW-NB15": "Normal", "CIC-IDS-2017": "BENIGN", "CICIoT23": "BenignTraffic"}


def _sum_cols(df, cols):
    return df[cols].sum(axis=1).values.astype(float)


def build_common_repr(dataset, X, y):
    spec = COMMON_COLS[dataset]
    feats = {}
    for name, cols in spec.items():
        cols = [c for c in cols if c in X.columns]
        feats[name] = _sum_cols(X, cols) if cols else np.zeros(len(X))
    common = pd.DataFrame(feats)
    y_bin = (np.asarray(y) != BENIGN_LABEL[dataset]).astype(int)
    return common, y_bin


def load_all():
    out = {}
    X, y, feat = mc.load_unsw_multiclass()
    out["UNSW-NB15"] = build_common_repr("UNSW-NB15", X, y)
    X, y, feat = mc.load_cic_multiclass()
    out["CIC-IDS-2017"] = build_common_repr("CIC-IDS-2017", X, y)
    X, y, feat = mc.load_ciciot23_multiclass()
    out["CICIoT23"] = build_common_repr("CICIoT23", X, y)
    return out


def _metrics(y_true, pred, proba):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "pr_auc": average_precision_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier_calibration": brier_score_loss(y_true, proba),
    }


def main():
    data = load_all()
    splits = {}
    for name, (common, y_bin) in data.items():
        n = min(15000, len(y_bin))
        idx = np.random.default_rng(SEED).choice(len(y_bin), n, replace=False)
        c, yb = common.iloc[idx].reset_index(drop=True), y_bin[idx]
        Xtr_full, Xte, ytr_full, yte = train_test_split(c, yb, test_size=0.2, stratify=yb, random_state=SEED)
        scaler = MinMaxScaler().fit(Xtr_full)
        splits[name] = {
            "Xtr": scaler.transform(Xtr_full), "ytr": ytr_full,
            "Xte": scaler.transform(Xte), "yte": yte,
            "scaler": scaler, "Xtr_raw": Xtr_full,
        }

    rows = []
    for source in splits:
        s = splits[source]
        clf = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                 eval_metric="logloss", n_jobs=4, verbosity=0, random_state=SEED)
        clf.fit(s["Xtr"], s["ytr"])
        proba_in = clf.predict_proba(s["Xte"])[:, 1]
        in_domain = _metrics(s["yte"], (proba_in >= 0.5).astype(int), proba_in)
        rows.append({"source": source, "target": source, "mode": "in_domain", **in_domain})
        print(f"[in-domain] {source}: macro_f1={in_domain['macro_f1']:.4f} bal_acc={in_domain['balanced_accuracy']:.4f}")

        for target in splits:
            if target == source:
                continue
            t = splits[target]
            Xte_t = t["scaler"].transform(pd.DataFrame(t["Xte"], columns=["duration", "rate", "total_bytes"])) \
                if False else t["Xte"]  # already scaled with the TARGET's own scaler (its own domain's units)
            proba_zs = clf.predict_proba(Xte_t)[:, 1]
            zs = _metrics(t["yte"], (proba_zs >= 0.5).astype(int), proba_zs)
            drop = in_domain["macro_f1"] - zs["macro_f1"]
            rows.append({"source": source, "target": target, "mode": "zero_shot",
                         "macro_f1_drop_vs_source_in_domain": drop, **zs})
            print(f"  [zero-shot {source} -> {target}] macro_f1={zs['macro_f1']:.4f} "
                  f"(drop vs. {source} in-domain: {drop:+.4f})")

            n_fs = min(FEW_SHOT_N, len(t["ytr"]))
            rng = np.random.default_rng(SEED)
            fs_idx = rng.choice(len(t["ytr"]), n_fs, replace=False)
            clf_fs = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                        eval_metric="logloss", n_jobs=4, verbosity=0, random_state=SEED)
            X_fs = np.vstack([s["Xtr"], t["Xtr"][fs_idx]])
            y_fs = np.concatenate([s["ytr"], t["ytr"][fs_idx]])
            clf_fs.fit(X_fs, y_fs)
            proba_fewshot = clf_fs.predict_proba(t["Xte"])[:, 1]
            fewshot = _metrics(t["yte"], (proba_fewshot >= 0.5).astype(int), proba_fewshot)
            rows.append({"source": source, "target": target, "mode": f"few_shot_n{n_fs}", **fewshot})
            print(f"  [few-shot(n={n_fs}) {source} -> {target}] macro_f1={fewshot['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "cross_dataset_transfer.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'cross_dataset_transfer.csv'}")


if __name__ == "__main__":
    main()
