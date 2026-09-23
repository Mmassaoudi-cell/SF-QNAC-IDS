"""
Orchestrates the full experiment grid: 3 datasets x {Protocol A, and Protocol
B where it exists} x 3 seeds x (7 ablation rungs + 5 additional benchmarks;
rung 1 already IS the XGBoost-default benchmark, so it is not duplicated).

Writes one row per run to results/raw_results.csv incrementally (so partial
progress survives an interruption), plus a manifest of exactly what ran.
"""
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import data_pipeline as dp
import ablation_ladder as al
import benchmarks as bm

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "raw_results.csv"
OUT_JSONL = RESULTS_DIR / "raw_results.jsonl"

SEEDS = [42, 43, 44, 45, 46, 47, 48]  # Phase 2: extended from 3 to 7 seeds

DATASET_PROTOCOLS = [
    ("UNSW-NB15", "A", lambda seed: dp.get_protocol_a(dp.load_unsw_raw, seed)),
    ("CIC-IDS-2017", "A", dp.get_cic_ids2017_protocol_a),
    ("CIC-IDS-2017", "B", dp.get_cic_ids2017_protocol_b),
    ("CICIoT23", "A", dp.get_ciciot23_protocol_a),
    ("CICIoT23", "B", dp.get_ciciot23_protocol_b),
]

EXTRA_BENCHMARKS = {k: v for k, v in bm.BENCHMARK_RUNNERS.items() if k != "1_xgboost_default"}


def _json_safe(v):
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def append_row(row):
    """JSONL append -- schema-safe by construction (unlike CSV-append, which
    corrupts when different methods' result dicts have different columns)."""
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps(_json_safe(row)) + "\n")


def flatten(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (dict,)):
            out[k] = json.dumps(_json_safe(v))
        elif isinstance(v, (list, np.ndarray)):
            lv = list(v)
            out[k] = json.dumps(_json_safe(lv)) if len(lv) < 50 else f"len={len(lv)}"
        else:
            out[k] = _json_safe(v)
    return out


def rebuild_csv_from_jsonl():
    """Regenerate raw_results.csv from the schema-safe JSONL log -- pandas
    aligns the union of all columns automatically (NaN-fills missing ones),
    which a raw CSV append cannot do once different methods contribute
    different columns."""
    if not OUT_JSONL.exists():
        return
    records = [json.loads(line) for line in open(OUT_JSONL) if line.strip()]
    pd.DataFrame(records).to_csv(OUT_CSV, index=False)


def main():
    done_keys = set()
    if OUT_JSONL.exists():
        for line in open(OUT_JSONL):
            if not line.strip():
                continue
            r = json.loads(line)
            done_keys.add((r.get("dataset"), r.get("protocol"), int(r.get("seed")), r.get("method")))
        print(f"Resuming: {len(done_keys)} runs already recorded.")

    for dataset, protocol, loader in DATASET_PROTOCOLS:
        for seed in SEEDS:
            t_split0 = time.time()
            split = loader(seed)
            dp.save_split_manifest(dataset, protocol, seed, split)
            print(f"\n[{dataset}/{protocol}/seed={seed}] split loaded in {time.time()-t_split0:.1f}s "
                  f"(n_train={len(split['y_train'])}, n_test={len(split['y_test'])})")

            all_methods = list(al.RUNG_FUNCS.items()) + [
                (name, (lambda split, seed, fn=fn: fn(
                    split["X_train"], split["y_train"], split["X_val"], split["y_val"],
                    split["X_test"], split["y_test"], seed=seed)))
                for name, fn in EXTRA_BENCHMARKS.items()
            ]

            for method_name, fn in all_methods:
                key = (dataset, protocol, seed, method_name)
                if key in done_keys:
                    print(f"  [skip, already done] {method_name}")
                    continue
                t0 = time.time()
                try:
                    result = fn(split, seed)
                    result["status"] = "ok"
                except Exception as e:
                    result = {"method": method_name, "status": "error", "error": str(e),
                               "traceback": traceback.format_exc()[-2000:]}
                    print(f"  [ERROR] {method_name}: {e}")
                result["dataset"] = dataset
                result["protocol"] = protocol
                result["seed"] = seed
                result["run_wall_time_sec"] = round(time.time() - t0, 2)
                append_row(result)
                f1 = result.get("test_f1_macro", float("nan"))
                print(f"  {method_name:38s} f1_macro={f1 if isinstance(f1,str) else (f'{f1:.4f}' if f1==f1 else 'NA')} "
                      f"time={result['run_wall_time_sec']:.1f}s")

            rebuild_csv_from_jsonl()  # refresh the CSV after every dataset/protocol/seed block

    rebuild_csv_from_jsonl()
    print(f"\nAll done. Results in {OUT_CSV}")


if __name__ == "__main__":
    main()
