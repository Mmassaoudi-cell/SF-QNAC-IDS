"""
Phase 2: tables + figures for noise robustness, shot sensitivity, qubit/depth
sensitivity, multiclass evaluation, and cross-dataset transfer -- all read
directly from the CSVs/JSON produced by noise_experiment.py,
shot_sensitivity.py, qubit_depth_sensitivity.py, multiclass_eval.py, and
cross_dataset_transfer.py. No numbers are hand-typed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

C_PROP = "#0E7C7B"
C_BASE = "#8B96A5"
C_MID = "#4B7BA6"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.25,
})


def escape_latex(s):
    return str(s).replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def df_to_latex_wide(df, caption, label):
    df = df.copy()
    df.columns = [escape_latex(c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(escape_latex)
    body = df.to_latex(index=False, escape=False, caption=escape_latex(caption),
                        label=label, float_format="%.4f", na_rep="--")
    body = body.replace("\\begin{table}", "\\begin{table*}").replace("\\end{table}", "\\end{table*}")
    body = body.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}")
    body = body.replace("\\end{tabular}", "\\end{tabular}}")
    return body


# --------------------------------------------------------------------------
# 1. Noise robustness table + figure
# --------------------------------------------------------------------------
def noise_outputs():
    df = pd.read_csv(RESULTS_DIR / "noise_robustness.csv")
    g = df.groupby(["condition", "mitigation"]).agg(
        mean_KL=("kl_div_from_ideal", "mean"),
        action_change_rate=("action_changed", "mean"),
        mean_downstream_f1=("downstream_test_f1_macro", "mean"),
    ).reset_index()
    order = [("ideal", "none"), ("finite_shot", "none"), ("depolarizing", "none"),
             ("readout", "none"), ("readout", "readout"), ("combined", "none"),
             ("combined", "readout"), ("combined", "zne")]
    g["order"] = g.apply(lambda r: order.index((r["condition"], r["mitigation"]))
                          if (r["condition"], r["mitigation"]) in order else 99, axis=1)
    g = g.sort_values("order").drop(columns="order")
    g.columns = ["Condition", "Mitigation", "Mean KL-div. from ideal", "Action-change rate", "Mean downstream test F1"]
    g.to_csv(TABLES_DIR / "noise_robustness_summary.csv", index=False)
    with open(TABLES_DIR / "noise_robustness_summary.tex", "w") as f:
        f.write(df_to_latex_wide(g, "NISQ-like noise robustness of the trained SF-QNAC-IDS policy (mean over 3 datasets x 2 seeds).",
                                  "tab:noise"))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    labels = [f"{c}\n({m})" if m != "none" else c for c, m in zip(g["Condition"], g["Mitigation"])]
    colors = [C_PROP if m == "none" else C_MID for m in g["Mitigation"]]
    ax.bar(range(len(g)), g["Mean KL-div. from ideal"], color=colors)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean KL-divergence from ideal policy")
    ax.set_title("Policy divergence under NISQ-like noise, with/without mitigation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_noise_robustness.png", bbox_inches="tight")
    plt.close(fig)
    return g


# --------------------------------------------------------------------------
# 2. Shot sensitivity figure
# --------------------------------------------------------------------------
def shot_sensitivity_outputs():
    df = pd.read_csv(RESULTS_DIR / "shot_sensitivity.csv")
    g = df.groupby("shots").agg(f1=("test_f1_macro", "mean"), f1_std=("test_f1_macro", "std"),
                                 kl=("kl_div_from_ideal_policy", "mean")).reset_index()
    g.to_csv(TABLES_DIR / "shot_sensitivity_summary.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.errorbar(g["shots"], g["f1"] * 100, yerr=g["f1_std"] * 100, marker="o", color=C_PROP, capsize=3)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Shots per policy read-out (log2)")
    ax1.set_ylabel("Test Macro-F1 (%)")
    ax1.set_title("(a) Accuracy vs. shot count")

    ax2.plot(g["shots"], g["kl"], marker="s", color=C_MID)
    ax2.set_xscale("log", base=2); ax2.set_yscale("log")
    ax2.set_xlabel("Shots per policy read-out (log2)")
    ax2.set_ylabel("Mean KL-div. from ideal (0-shot) policy, log")
    ax2.set_title("(b) Policy fidelity vs. shot count")
    fig.suptitle("Shot-sensitivity sweep: 64-8192 shots")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_shot_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    return g


# --------------------------------------------------------------------------
# 3. Qubit/depth sensitivity heatmap
# --------------------------------------------------------------------------
def qubit_depth_outputs():
    df = pd.read_csv(RESULTS_DIR / "qubit_depth_sensitivity.csv")
    g = df.groupby(["n_qubits", "layers"]).agg(f1=("test_f1_macro", "mean")).reset_index()
    pivot = g.pivot(index="n_qubits", columns="layers", values="f1") * 100
    pivot.to_csv(TABLES_DIR / "qubit_depth_sensitivity_summary.csv")

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Layers (depth)"); ax.set_ylabel("Qubits")
    ax.set_title("Test Macro-F1 (%) vs. qubits x layers\n(mean over 2 datasets, 1 seed)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center",
                     color="white" if pivot.values[i, j] < pivot.values.mean() else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Macro-F1 (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_qubit_depth_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    return pivot


# --------------------------------------------------------------------------
# 4. Multiclass table + per-class figure
# --------------------------------------------------------------------------
def multiclass_outputs():
    with open(RESULTS_DIR / "multiclass_results.json") as f:
        res = json.load(f)
    rows = []
    for ds, r in res.items():
        rows.append({
            "Dataset": ds, "Classes evaluated": f"{r['n_classes_evaluated']}/{r['n_classes_full_data']}",
            "Accuracy": r["accuracy"], "Balanced accuracy": r["balanced_accuracy"],
            "Macro-F1": r["macro_f1"], "Weighted-F1": r["weighted_f1"],
            "Macro precision": r["macro_precision"], "Macro recall": r["macro_recall"],
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TABLES_DIR / "multiclass_summary.csv", index=False)
    with open(TABLES_DIR / "multiclass_summary.tex", "w") as f:
        f.write(df_to_latex_wide(tbl, "Multiclass evaluation: overall accuracy vs. class-balanced metrics.", "tab:multiclass"))

    # per-class F1 bar for the dataset with the most classes AND worst balanced accuracy (UNSW-NB15)
    ds = "UNSW-NB15"
    report = res[ds]["per_class_report"]
    classes = [c for c in report if c not in ("accuracy", "macro avg", "weighted avg")]
    f1s = [report[c]["f1-score"] for c in classes]
    supports = [report[c]["support"] for c in classes]
    order = np.argsort(supports)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(range(len(classes)), [f1s[i] for i in order], color=C_PROP)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([classes[i] for i in order], rotation=40, ha="right", fontsize=8)
    for i, idx in enumerate(order):
        ax.annotate(f"n={int(supports[idx])}", (i, f1s[idx] + 0.02), ha="center", fontsize=6.5, color="#555")
    ax.set_ylabel("Per-class F1")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Per-class F1 vs. class support -- {ds} multiclass\n(classes ordered by test-set support, most common first)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_perclass_f1.png", bbox_inches="tight")
    plt.close(fig)
    return tbl


# --------------------------------------------------------------------------
# 5. Cross-dataset transfer table
# --------------------------------------------------------------------------
def transfer_outputs():
    df = pd.read_csv(RESULTS_DIR / "cross_dataset_transfer.csv")
    keep = ["source", "target", "mode", "accuracy", "balanced_accuracy", "macro_f1", "pr_auc", "roc_auc"]
    tbl = df[keep].copy()
    tbl.to_csv(TABLES_DIR / "cross_dataset_transfer_summary.csv", index=False)
    with open(TABLES_DIR / "cross_dataset_transfer_summary.tex", "w") as f:
        f.write(df_to_latex_wide(tbl, "Cross-dataset transfer: in-domain vs. zero-shot vs. few-shot (n=300), common 3-feature representation.",
                                  "tab:transfer"))
    return tbl


def main():
    print("Noise:\n", noise_outputs())
    print("\nShot sensitivity:\n", shot_sensitivity_outputs())
    print("\nQubit/depth:\n", qubit_depth_outputs())
    print("\nMulticlass:\n", multiclass_outputs())
    print("\nTransfer:\n", transfer_outputs())
    print("\nAll Phase-2 tables/figures written.")


if __name__ == "__main__":
    main()
