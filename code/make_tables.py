"""
Generates the required tables (CSV provenance + LaTeX snippets) straight
from results/raw_results.csv. Every number in the manuscript's tables must
trace back to a row in this CSV -- no numbers are retyped by hand.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"
TABLES_DIR.mkdir(exist_ok=True)

ABLATION_ORDER = [
    "1_xgboost_default", "2_sho_xgboost", "3_qrl_sho_xgboost_reproduction",
    "4_classical_actor_critic", "5_vqc_actor_critic_ordinary_grad",
    "6_qnac_fixed_shots", "7_sf_qnac_ids",
]
ABLATION_LABELS = {
    "1_xgboost_default": "XGBoost",
    "2_sho_xgboost": "SHO-XGBoost",
    "3_qrl_sho_xgboost_reproduction": "QRL-SHO-XGBoost (repro.)",
    "4_classical_actor_critic": "Classical actor-critic",
    "5_vqc_actor_critic_ordinary_grad": "VQC actor-critic (ord. grad)",
    "6_qnac_fixed_shots": "QNAC (fixed shots)",
    "7_sf_qnac_ids": "SF-QNAC-IDS (full)",
}
BENCH_ORDER = ["bench_logreg", "bench_random_forest", "bench_lightgbm", "bench_optuna_xgboost",
               "bench_quantum_kernel_classifier"]
BENCH_LABELS = {
    "bench_logreg": "Logistic Regression", "bench_random_forest": "Random Forest",
    "bench_lightgbm": "LightGBM", "bench_optuna_xgboost": "Optuna-XGBoost (25 trials)",
    "bench_quantum_kernel_classifier": "Quantum-kernel classifier",
}


def load():
    df = pd.read_csv(RESULTS_DIR / "raw_results.csv")
    return df[df["status"] == "ok"].copy()


def agg(df, group_cols, value_cols):
    g = df.groupby(group_cols)[value_cols].agg(["mean", "std"])
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    return g.reset_index()


def fmt(mean, std, pct=True, decimals=2):
    if pd.isna(mean):
        return "--"
    m = mean * 100 if pct else mean
    s = (std * 100 if pct else std) if not pd.isna(std) else 0.0
    return f"{m:.{decimals}f}$\\pm${s:.{decimals}f}"


def build_ablation_table(df):
    value_cols = ["n_qubits", "circuit_depth", "total_shots", "test_f1_macro", "test_pr_auc",
                  "test_fpr", "episodes_to_convergence", "total_circuit_evals",
                  "total_full_trainings", "wall_time_total_sec"]
    for c in value_cols:
        if c not in df.columns:
            df[c] = np.nan
    rows = []
    for method in ABLATION_ORDER:
        sub = df[df["method"] == method]
        if len(sub) == 0:
            continue
        rows.append({
            "Variant": ABLATION_LABELS[method],
            "Qubits": sub["n_qubits"].mean(),
            "Depth": sub["circuit_depth"].mean() if "circuit_depth" in sub else np.nan,
            "Shots (mean)": sub["total_shots"].mean(),
            "Macro-F1 (mean$\pm$std %)": fmt(sub["test_f1_macro"].mean(), sub["test_f1_macro"].std()),
            "PR-AUC (mean$\pm$std %)": fmt(sub["test_pr_auc"].mean(), sub["test_pr_auc"].std()),
            "FPR (mean %)": fmt(sub["test_fpr"].mean(), sub["test_fpr"].std()),
            "Episodes-to-conv (mean)": sub["episodes_to_convergence"].mean(),
            "Quantum evals (mean)": sub["total_circuit_evals"].mean(),
            "Full trainings (mean)": sub["total_full_trainings"].mean(),
            "Train time s (mean)": sub["wall_time_total_sec"].mean(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "ablation_ladder.csv", index=False)
    return out


def build_main_results_table(df):
    df["ds_proto"] = df["dataset"] + "-" + df["protocol"]
    pivot_f1 = df.pivot_table(index="method", columns="ds_proto", values="test_f1_macro", aggfunc="mean")
    pivot_fpr = df.groupby("method")["test_fpr"].mean()
    pivot_time = df.groupby("method")["wall_time_total_sec"].mean()
    pivot_shots = df.groupby("method")["total_shots"].mean() if "total_shots" in df.columns else None

    order = ABLATION_ORDER + BENCH_ORDER
    labels = {**ABLATION_LABELS, **BENCH_LABELS}
    rows = []
    for method in order:
        if method not in pivot_f1.index:
            continue
        row = {"Method": labels[method]}
        for col in pivot_f1.columns:
            row[col] = pivot_f1.loc[method, col] * 100 if not pd.isna(pivot_f1.loc[method, col]) else np.nan
        row["Avg FPR (%)"] = pivot_fpr.get(method, np.nan) * 100
        row["Train Time (s)"] = pivot_time.get(method, np.nan)
        row["Quantum Shots"] = pivot_shots.get(method, np.nan) if pivot_shots is not None else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "main_results.csv", index=False)
    return out


def _circuit_gate_counts():
    """Real, structurally-computed (not hand-counted) gate counts for each
    quantum circuit family, via the same Circuit.depth()/gate-count methods
    used throughout (see quantum_sim.py)."""
    import numpy as np
    from quantum_sim import baseline_circuit, squire_circuit
    from qnac_core import QuantumActor
    feats8 = np.random.default_rng(0).uniform(0, np.pi, 8)
    b = baseline_circuit(feats8)
    actor = QuantumActor(state_dim=12, n_actions=19, n_qubits=4, layers=2, seed=0)
    s = squire_circuit(feats8)
    astats = actor.circuit_stats()
    return {
        "3_qrl_sho_xgboost_reproduction": f"{b.one_qubit_gate_count()}/{b.two_qubit_gate_count()}",
        "5_vqc_actor_critic_ordinary_grad": f"{astats['gates_1q']}/{astats['gates_2q']}",
        "6_qnac_fixed_shots": f"{astats['gates_1q']}/{astats['gates_2q']}",
        "7_sf_qnac_ids": f"{astats['gates_1q']}/{astats['gates_2q']}",
        "bench_quantum_kernel_classifier": f"{s.one_qubit_gate_count()}/{s.two_qubit_gate_count()}",
    }


def build_quantum_efficiency_table(df):
    quantum_methods = ["3_qrl_sho_xgboost_reproduction", "5_vqc_actor_critic_ordinary_grad",
                        "6_qnac_fixed_shots", "7_sf_qnac_ids", "bench_quantum_kernel_classifier"]
    gate_counts = _circuit_gate_counts()
    rows = []
    for method in quantum_methods:
        sub = df[df["method"] == method]
        if len(sub) == 0:
            continue
        rows.append({
            "Method": {**ABLATION_LABELS, **BENCH_LABELS}.get(method, method),
            "Qubits": sub["n_qubits"].mean() if "n_qubits" in sub else np.nan,
            "1Q/2Q gates": gate_counts.get(method, "--"),
            "Shots (mean)": sub["total_shots"].mean() if "total_shots" in sub else np.nan,
            "Circuit calls (mean)": sub["total_circuit_evals"].mean() if "total_circuit_evals" in sub else np.nan,
            "Episodes (mean)": sub["episodes_to_convergence"].mean() if "episodes_to_convergence" in sub else np.nan,
            "Runtime s (mean)": sub["wall_time_total_sec"].mean(),
            "Macro-F1 (mean %)": sub["test_f1_macro"].mean() * 100,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "quantum_efficiency.csv", index=False)
    return out


def _sci(x):
    if pd.isna(x):
        return "--"
    if isinstance(x, (int, float, np.floating, np.integer)) and abs(x) >= 100000:
        return f"{x:.2e}"
    return x


def _escape_header(name):
    # column headers are plain strings (unlike value cells, which may
    # intentionally contain our own LaTeX math like $\pm$) -- raw '%', '_',
    # '#' in a header is a LaTeX comment/subscript character and corrupts
    # everything after it on that line if left unescaped.
    return str(name).replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def df_to_latex(df, caption, label, wide=False):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].apply(_sci)
    df.columns = [_escape_header(c) for c in df.columns]
    import re
    caption = re.sub(r"(?<!\\)%", r"\\%", caption)
    body = df.to_latex(index=False, escape=False, caption=caption, label=label,
                        float_format="%.3f", na_rep="--")
    if wide:
        body = body.replace("\\begin{table}", "\\begin{table*}").replace("\\end{table}", "\\end{table*}")
        # \small alone still overflows the double-column text width once a
        # table has 8+ numeric columns; force-fit with \resizebox instead.
        body = body.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}")
        body = body.replace("\\end{tabular}", "\\end{tabular}}")
    return body


def build_stats_summary_table():
    p = TABLES_DIR / "stats_headline_comparison.csv"
    if not p.exists():
        return None
    stats = pd.read_csv(p)
    f1 = stats[stats["metric"] == "test_f1_macro"].copy()
    f1["dataset/protocol"] = f1["dataset"] + "-" + f1["protocol"]
    out = f1[["dataset/protocol", "mean_A_qrl_sho_xgboost_repro", "mean_B_sf_qnac_ids",
              "mean_diff_B_minus_A", "diff_ci_lo_95", "diff_ci_hi_95",
              "paired_t_pvalue", "paired_t_pvalue_holm", "wilcoxon_pvalue"]].copy()
    out.columns = ["Dataset-Protocol", "F1 QRL-SHO-XGB (repro.)", "F1 SF-QNAC-IDS",
                   "Mean diff.", "95% CI lo", "95% CI hi", "paired-t p", "Holm-adj. p", "Wilcoxon p"]
    for c in out.columns[1:]:
        out[c] = out[c].astype(float)
    out.to_csv(TABLES_DIR / "stats_headline_comparison_summary.csv", index=False)
    return out


def main():
    df = load()
    ab = build_ablation_table(df)
    main_res = build_main_results_table(df)
    qeff = build_quantum_efficiency_table(df)
    stats_summary = build_stats_summary_table()

    with open(TABLES_DIR / "ablation_ladder.tex", "w") as f:
        f.write(df_to_latex(ab, "Ablation ladder: isolating the source of any improvement.", "tab:ablation", wide=True))
    with open(TABLES_DIR / "main_results.tex", "w") as f:
        f.write(df_to_latex(main_res, "Main results across datasets/protocols (test Macro-F1 \\%, mean of 3 seeds; Avg FPR / Train Time / Quantum Shots pooled means).", "tab:main", wide=True))
    with open(TABLES_DIR / "quantum_efficiency.tex", "w") as f:
        f.write(df_to_latex(qeff, "Quantum resource usage vs. detection performance (means across dataset/protocol/seed cells).", "tab:qeff", wide=True))
    if stats_summary is not None:
        with open(TABLES_DIR / "stats_headline_comparison_summary.tex", "w") as f:
            f.write(df_to_latex(stats_summary,
                                 "SF-QNAC-IDS vs. QRL-SHO-XGBoost reproduction, paired by seed (n=3), test Macro-F1.",
                                 "tab:stats", wide=True))

    print("Ablation ladder:\n", ab.to_string(index=False))
    print("\nMain results:\n", main_res.to_string(index=False))
    print("\nQuantum efficiency:\n", qeff.to_string(index=False))
    if stats_summary is not None:
        print("\nStats summary:\n", stats_summary.to_string(index=False))


if __name__ == "__main__":
    main()
