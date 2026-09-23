"""
Leakage-safe data pipeline for UNSW-NB15, CIC-IDS-2017, CICIoT23.

Protocol A: stratified 70/15/15 (paper-compatible).
Protocol B: a REAL disjoint split per dataset (day-disjoint for CIC-IDS-2017,
official train/test/validation files for CICIoT23; UNSW-NB15 has no native
disjoint field so Protocol B is not claimed for it -- see get_protocol_b()).

All scaling/encoding is fit on TRAIN only and applied to val/test, for both
protocols. A duplicate/leakage audit is run and saved for every split.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(r"C:\Users\MMASSAOUDI\Desktop\Data")
SPLIT_DIR = Path(__file__).resolve().parent.parent / "splits"
SPLIT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

WORKING_N = 20000  # stratified working subsample size per dataset, this phase

# Repeatedly re-reading CIC-IDS-2017's ~885MB of day-files and CICIoT23's
# multi-GB files once PER SEED (3x for Protocol A, 3x for Protocol B) was
# causing memory pressure and unexplained background-process termination
# during the full grid run. Each large source is now pooled ONCE, with a
# fixed internal seed (independent of the experiment seed -- this is also
# methodologically cleaner: every experiment seed subsamples/splits the same
# underlying population instead of each seed also re-sampling a different
# population), cached to disk, and re-used for every experiment seed.
POOL_SEED = 0


def _cache_path(name):
    return CACHE_DIR / f"{name}.parquet"


def _cached_pool(name, build_fn):
    p = _cache_path(name)
    if p.exists():
        df = pd.read_parquet(p)
        y = df["__y__"].values
        X = df.drop(columns=["__y__"])
        return X, y
    X, y = build_fn()
    out = X.copy()
    out["__y__"] = y
    out.to_parquet(p, index=False)
    return X, y


def _row_hash(df: pd.DataFrame) -> pd.Series:
    """Exact-duplicate detector: hash of the row's feature values (label excluded)."""
    return pd.util.hash_pandas_object(df, index=False)


def _encode_frame(X: pd.DataFrame):
    X = X.copy()
    for c in X.select_dtypes(include="object").columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return X.fillna(0)


def _duplicate_audit(name, X_train, X_other, other_name):
    h_train = set(_row_hash(X_train).tolist())
    h_other = _row_hash(X_other)
    cross_dupe_mask = h_other.isin(h_train)
    audit = {
        "dataset": name,
        "n_train": int(len(X_train)),
        f"n_{other_name}": int(len(X_other)),
        "exact_duplicates_within_train": int(X_train.duplicated().sum()),
        f"exact_duplicates_within_{other_name}": int(X_other.duplicated().sum()),
        f"cross_split_exact_duplicates_train_vs_{other_name}": int(cross_dupe_mask.sum()),
        f"cross_split_duplicate_rate_{other_name}": float(cross_dupe_mask.mean()),
    }
    return audit


def stratified_subsample(X, y, n, seed):
    if n is None or n >= len(y):
        return X, y
    vals, counts = np.unique(y, return_counts=True)
    if (counts < 2).any() or len(vals) < 2:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
        return X.iloc[idx].reset_index(drop=True), y[idx]
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def deduplicate(X: pd.DataFrame, y: np.ndarray):
    """Drop exact-duplicate feature rows (label excluded from the hash) before
    any split is made, so duplicate leakage cannot enter train/val/test via
    the split itself. Returns (X, y, n_dropped)."""
    h = _row_hash(X)
    keep = ~h.duplicated()
    n_dropped = int((~keep).sum())
    return X.loc[keep].reset_index(drop=True), y[keep.values], n_dropped


def read_csv_stratified_by_chunks(path, label_fn, target_n, seed, chunksize=400000,
                                   feature_drop_cols=None, max_chunks=None):
    """Read a (possibly time-ordered) CSV in chunks and take a proportional
    sample from EACH chunk, so the result is representative of the whole file
    rather than biased toward whatever class dominates the first rows -- this
    matters for capture-ordered files like CIC-IDS-2017's per-day CSVs."""
    rng = np.random.default_rng(seed)
    parts = []
    n_seen = 0
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
        if feature_drop_cols:
            chunk = chunk.rename(columns=lambda c: c.strip())
        y_chunk = label_fn(chunk)
        n_seen += len(chunk)
        take = max(1, int(target_n * len(chunk) / max(target_n * 6, chunksize)))
        take = min(take, len(chunk))
        idx = rng.choice(len(chunk), size=take, replace=False)
        parts.append((chunk.iloc[idx].reset_index(drop=True), y_chunk[idx]))
        if max_chunks and i + 1 >= max_chunks:
            break
    X = pd.concat([p[0] for p in parts], ignore_index=True)
    y = np.concatenate([p[1] for p in parts])
    return X, y


# --------------------------------------------------------------------------
# UNSW-NB15
# --------------------------------------------------------------------------
def load_unsw_raw():
    tr = pd.read_csv(DATA_DIR / "UNSW_NB15" / "UNSW_NB15_training-set.csv")
    te = pd.read_csv(DATA_DIR / "UNSW_NB15" / "UNSW_NB15_testing-set.csv")
    df = pd.concat([tr, te], ignore_index=True)
    y = df["label"].astype(int).values
    y_multi = df["attack_cat"].fillna("Normal").astype(str).values
    X = df.drop(columns=["id", "label", "attack_cat"])
    X = _encode_frame(X)
    return X, y, y_multi, list(X.columns)


def get_protocol_a(loader, seed, working_n=WORKING_N):
    X, y, y_multi, feat_names = loader()
    X, y, n_dropped_raw = deduplicate(X, y)
    Xs, ys = stratified_subsample(X, y, working_n, seed)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        Xs, ys, test_size=0.15, stratify=ys, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, stratify=y_train_full, random_state=seed)
    audit = _duplicate_audit("protocol_a", X_train, X_test, "test")
    audit["exact_duplicates_dropped_before_split_full_dataset"] = n_dropped_raw
    audit["full_dataset_size_before_dedup"] = int(len(y) + n_dropped_raw)
    return {
        "X_train": X_train.values.astype(float), "y_train": y_train,
        "X_val": X_val.values.astype(float), "y_val": y_val,
        "X_test": X_test.values.astype(float), "y_test": y_test,
        "feature_names": feat_names, "audit": audit,
    }


# --------------------------------------------------------------------------
# CIC-IDS-2017 (day-disjoint => real Protocol B)
# --------------------------------------------------------------------------
CIC_DIR = DATA_DIR / "CIC-IDS- 2017"
CIC_FILES = {
    "monday": "Monday-WorkingHours.pcap_ISCX.csv",           # 100% benign
    "tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",         # brute force
    "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",     # DoS/Heartbleed
    "thu_morning": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "thu_afternoon": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "fri_morning": "Friday-WorkingHours-Morning.pcap_ISCX.csv",   # botnet
    "fri_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "fri_portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
}


def _cic_label_fn(chunk):
    col = "Label" if "Label" in chunk.columns else [c for c in chunk.columns if c.strip() == "Label"][0]
    return (chunk[col].astype(str).str.strip() != "BENIGN").astype(int).values


CIC_DAY_POOL_N = 6000  # per-day cached-pool size (with margin over any single caller's need)


def _build_cic_day_pool(day_key):
    path = CIC_DIR / CIC_FILES[day_key]
    X, y = read_csv_stratified_by_chunks(path, _cic_label_fn, CIC_DAY_POOL_N, POOL_SEED, chunksize=300000)
    X.columns = [c.strip() for c in X.columns]
    X = X.drop(columns=["Label"])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = _encode_frame(X)
    return X, y


def _load_cic_day(day_key, target_n, seed):
    """Position-unbiased sample from one day's capture file, read ONCE (cached
    to disk, fixed POOL_SEED) and re-sampled in-memory per experiment seed --
    the original per-seed full-file re-read caused memory pressure and
    unexplained process termination during the full grid run."""
    Xpool, ypool = _cached_pool(f"cic2017_{day_key}", lambda: _build_cic_day_pool(day_key))
    return stratified_subsample(Xpool, ypool, min(target_n, len(ypool)), seed)


def get_cic_ids2017_protocol_a(seed, working_n=WORKING_N):
    # pool a representative sample across all 8 days, then do the standard 70/15/15
    parts_X, parts_y = [], []
    per_day_cap = working_n // len(CIC_FILES) + 500
    for day in CIC_FILES:
        Xd, yd = _load_cic_day(day, per_day_cap, seed)
        parts_X.append(Xd); parts_y.append(yd)
    X = pd.concat(parts_X, ignore_index=True)
    y = np.concatenate(parts_y)
    X, y, n_dropped_raw = deduplicate(X, y)
    Xs, ys = stratified_subsample(X, y, working_n, seed)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        Xs, ys, test_size=0.15, stratify=ys, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, stratify=y_train_full, random_state=seed)
    audit = _duplicate_audit("cic2017_protocol_a", X_train, X_test, "test")
    audit["exact_duplicates_dropped_before_split_full_dataset"] = n_dropped_raw
    feat_names = list(X.columns)
    return {
        "X_train": X_train.values.astype(float), "y_train": y_train,
        "X_val": X_val.values.astype(float), "y_val": y_val,
        "X_test": X_test.values.astype(float), "y_test": y_test,
        "feature_names": feat_names, "audit": audit,
    }


def get_cic_ids2017_protocol_b(seed, working_n=WORKING_N):
    """REAL day-disjoint split: train/val on Mon-Thu captures, test on Friday
    captures (unseen capture sessions -- different day, different attack mix)."""
    train_days = ["monday", "tuesday", "wednesday", "thu_morning", "thu_afternoon"]
    test_days = ["fri_morning", "fri_ddos", "fri_portscan"]
    per_day_cap = int((working_n * 0.85) // len(train_days) + 500)
    tr_X, tr_y = [], []
    for day in train_days:
        Xd, yd = _load_cic_day(day, per_day_cap, seed)
        tr_X.append(Xd); tr_y.append(yd)
    X_train_full = pd.concat(tr_X, ignore_index=True)
    y_train_full = np.concatenate(tr_y)

    per_day_cap_test = int((working_n * 0.15) // len(test_days) + 300)
    te_X, te_y = [], []
    for day in test_days:
        Xd, yd = _load_cic_day(day, per_day_cap_test, seed)
        te_X.append(Xd); te_y.append(yd)
    X_test = pd.concat(te_X, ignore_index=True)
    y_test = np.concatenate(te_y)

    # align columns (day files can differ very slightly in dtype after encoding)
    common_cols = [c for c in X_train_full.columns if c in X_test.columns]
    X_train_full = X_train_full[common_cols]
    X_test = X_test[common_cols]
    X_train_full, y_train_full, n_drop_tr = deduplicate(X_train_full, y_train_full)
    X_test, y_test, n_drop_te = deduplicate(X_test, y_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, stratify=y_train_full, random_state=seed)
    audit = _duplicate_audit("cic2017_protocol_b_day_disjoint", X_train, X_test, "test")
    audit["exact_duplicates_dropped_before_split_train_side"] = n_drop_tr
    audit["exact_duplicates_dropped_before_split_test_side"] = n_drop_te
    return {
        "X_train": X_train.values.astype(float), "y_train": y_train,
        "X_val": X_val.values.astype(float), "y_val": y_val,
        "X_test": X_test.values.astype(float), "y_test": y_test,
        "feature_names": common_cols, "audit": audit,
        "protocol_b_kind": "day/capture-disjoint (train=Mon-Thu, test=Fri)",
    }


# --------------------------------------------------------------------------
# CICIoT23 (official split => real Protocol B)
# --------------------------------------------------------------------------
CICIOT_DIR = DATA_DIR / "CICIOT23" / "CICIOT23"


def _ciciot_label_fn(chunk):
    return (chunk["label"].astype(str) != "BenignTraffic").astype(int).values


CICIOT_PART_POOL_N = 18000  # cached-pool size per official file (with margin)


def _build_ciciot_part_pool(part, max_chunks=6, chunksize=300000):
    f = CICIOT_DIR / part / f"{part}.csv"
    X, y = read_csv_stratified_by_chunks(f, _ciciot_label_fn, CICIOT_PART_POOL_N, POOL_SEED,
                                          chunksize=chunksize, max_chunks=max_chunks)
    X = X.drop(columns=["label"])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = _encode_frame(X)
    return X, y


def _load_ciciot_part(part, target_n, seed):
    """Same fix as CIC-IDS-2017: read each official file ONCE (cached), then
    re-sample per experiment seed in-memory instead of re-reading the
    multi-GB source file for every seed."""
    Xpool, ypool = _cached_pool(f"ciciot23_{part}", lambda: _build_ciciot_part_pool(part))
    return stratified_subsample(Xpool, ypool, min(target_n, len(ypool)), seed)


def get_ciciot23_protocol_a(seed, working_n=WORKING_N):
    # pool train+test+validation official files, then re-split 70/15/15 (paper-compatible)
    Xtr, ytr = _load_ciciot_part("train", int(working_n * 0.7), seed)
    Xte, yte = _load_ciciot_part("test", int(working_n * 0.15), seed)
    Xva, yva = _load_ciciot_part("validation", int(working_n * 0.15), seed)
    common_cols = [c for c in Xtr.columns if c in Xte.columns and c in Xva.columns]
    X = pd.concat([Xtr[common_cols], Xte[common_cols], Xva[common_cols]], ignore_index=True)
    y = np.concatenate([ytr, yte, yva])
    X, y, n_dropped_raw = deduplicate(X, y)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, stratify=y_train_full, random_state=seed)
    audit = _duplicate_audit("ciciot23_protocol_a", X_train, X_test, "test")
    audit["exact_duplicates_dropped_before_split_full_dataset"] = n_dropped_raw
    return {
        "X_train": X_train.values.astype(float), "y_train": y_train,
        "X_val": X_val.values.astype(float), "y_val": y_val,
        "X_test": X_test.values.astype(float), "y_test": y_test,
        "feature_names": common_cols, "audit": audit,
    }


def get_ciciot23_protocol_b(seed, working_n=WORKING_N):
    """REAL official split: the dataset's own pre-made train/validation/test files
    (published by the dataset authors), used as-is instead of a custom re-split."""
    Xtr, ytr = _load_ciciot_part("train", int(working_n * 0.7), seed)
    Xva, yva = _load_ciciot_part("validation", int(working_n * 0.15), seed)
    Xte, yte = _load_ciciot_part("test", int(working_n * 0.15), seed)
    common_cols = [c for c in Xtr.columns if c in Xte.columns and c in Xva.columns]
    Xtr, Xva, Xte = Xtr[common_cols], Xva[common_cols], Xte[common_cols]
    Xtr, ytr, n_drop_tr = deduplicate(Xtr, ytr)
    Xte, yte, n_drop_te = deduplicate(Xte, yte)
    audit = _duplicate_audit("ciciot23_protocol_b_official_split", Xtr, Xte, "test")
    return {
        "X_train": Xtr.values.astype(float), "y_train": ytr,
        "X_val": Xva.values.astype(float), "y_val": yva,
        "X_test": Xte.values.astype(float), "y_test": yte,
        "feature_names": common_cols, "audit": audit,
        "protocol_b_kind": "official author-provided train/validation/test files",
    }


# --------------------------------------------------------------------------
DATASET_LOADERS = {
    "UNSW-NB15": {"A": lambda seed: get_protocol_a(load_unsw_raw, seed), "B": None},
    "CIC-IDS-2017": {"A": get_cic_ids2017_protocol_a, "B": get_cic_ids2017_protocol_b},
    "CICIoT23": {"A": get_ciciot23_protocol_a, "B": get_ciciot23_protocol_b},
}


def save_split_manifest(dataset, protocol, seed, split):
    meta = {
        "dataset": dataset, "protocol": protocol, "seed": seed,
        "n_train": len(split["y_train"]), "n_val": len(split["y_val"]), "n_test": len(split["y_test"]),
        "n_features": len(split["feature_names"]),
        "positive_rate_train": float(split["y_train"].mean()),
        "positive_rate_test": float(split["y_test"].mean()),
        "audit": split["audit"],
        "protocol_b_kind": split.get("protocol_b_kind"),
    }
    out = SPLIT_DIR / f"{dataset}_{protocol}_seed{seed}_manifest.json"
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


if __name__ == "__main__":
    for name, loaders in DATASET_LOADERS.items():
        for proto, fn in loaders.items():
            if fn is None:
                print(f"{name} Protocol {proto}: not applicable (no native disjoint field) -- skipped, documented")
                continue
            split = fn(seed=42)
            meta = save_split_manifest(name, proto, 42, split)
            print(f"{name} Protocol {proto}: {json.dumps(meta, indent=2)}")
