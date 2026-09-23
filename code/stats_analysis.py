"""
Statistical testing on raw_results.csv: paired comparison of SF-QNAC-IDS vs.
the QRL-SHO-XGBoost reproduction (the headline claim), across the 3 seeds,
per dataset/protocol, plus a Holm correction across the resulting family of
tests.

Honesty note (see FINAL_RESEARCH_SUMMARY.md): with only 3 seeds per cell
(this phase's agreed scope, vs. the fuller spec's 5-10), a paired Wilcoxon
signed-rank test cannot reach p<0.05 in principle for n=3 (the smallest
attainable two-sided p-value at n=3 is 0.25) -- so Wilcoxon results here are
reported for completeness but the paired t-test and the bootstrap CI (which
remain informative, if wide, at n=3) are the tests actually load-bearing
for any claim at this phase's scale.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"
TABLES_DIR.mkdir(exist_ok=True)

METHOD_A = "3_qrl_sho_xgboost_reproduction"
METHOD_B = "7_sf_qnac_ids"
METRICS = ["test_f1_macro", "test_pr_auc", "test_fpr", "minority_recall"]


def bootstrap_ci(diffs, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    boots = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    df = pd.read_csv(RESULTS_DIR / "raw_results.csv")
    df = df[df["status"] == "ok"]

    rows = []
    for (dataset, protocol), g in df.groupby(["dataset", "protocol"]):
        a = g[g["method"] == METHOD_A].sort_values("seed")
        b = g[g["method"] == METHOD_B].sort_values("seed")
        if len(a) == 0 or len(b) == 0:
            continue
        common_seeds = sorted(set(a["seed"]) & set(b["seed"]))
        a = a[a["seed"].isin(common_seeds)].sort_values("seed")
        b = b[b["seed"].isin(common_seeds)].sort_values("seed")
        if len(a) < 2:
            continue
        for metric in METRICS:
            va, vb = a[metric].values.astype(float), b[metric].values.astype(float)
            diffs = vb - va
            t_stat, t_p = stats.ttest_rel(vb, va) if len(va) > 1 else (float("nan"), float("nan"))
            try:
                w_stat, w_p = stats.wilcoxon(vb, va)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            lo, hi = bootstrap_ci(diffs) if len(diffs) > 1 else (float("nan"), float("nan"))
            rows.append({
                "dataset": dataset, "protocol": protocol, "metric": metric, "n_seeds": len(common_seeds),
                "mean_A_qrl_sho_xgboost_repro": va.mean(), "std_A": va.std(ddof=1) if len(va) > 1 else 0.0,
                "mean_B_sf_qnac_ids": vb.mean(), "std_B": vb.std(ddof=1) if len(vb) > 1 else 0.0,
                "mean_diff_B_minus_A": diffs.mean(), "diff_ci_lo_95": lo, "diff_ci_hi_95": hi,
                "paired_t_stat": t_stat, "paired_t_pvalue": t_p,
                "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_p,
            })

    res = pd.DataFrame(rows)
    if len(res):
        # Holm correction across the family of paired-t tests on the primary metric (f1_macro)
        primary = res[res["metric"] == "test_f1_macro"].copy()
        m = len(primary)
        order = np.argsort(primary["paired_t_pvalue"].values)
        holm_adj = np.empty(m)
        sorted_p = primary["paired_t_pvalue"].values[order]
        running_max = 0.0
        for rank, idx in enumerate(order):
            adj = (m - rank) * sorted_p[rank]
            running_max = max(running_max, adj)
            holm_adj[idx] = min(running_max, 1.0)
        primary["paired_t_pvalue_holm"] = holm_adj
        res = res.merge(primary[["dataset", "protocol", "paired_t_pvalue_holm"]],
                         on=["dataset", "protocol"], how="left")

    res.to_csv(TABLES_DIR / "stats_headline_comparison.csv", index=False)
    print(res.to_string(index=False))
    print(f"\nSaved {TABLES_DIR / 'stats_headline_comparison.csv'}")


if __name__ == "__main__":
    main()
