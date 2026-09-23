"""
Generates the figures this phase's scope actually supports from saved CSVs
(figures 1-2 are schematic architecture diagrams; 3-5 are data-driven and
read directly from results/raw_results.csv). The full spec's shot/qubit/
depth sensitivity sweeps and multiclass confusion-matrix figures are
Phase 2 items (not run this phase) and are not faked here with placeholder
numbers.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.25,
})

C_BASE = "#8B96A5"
C_PROP = "#0E7C7B"
C_MID = "#4B7BA6"

ABLATION_ORDER = [
    "1_xgboost_default", "2_sho_xgboost", "3_qrl_sho_xgboost_reproduction",
    "4_classical_actor_critic", "5_vqc_actor_critic_ordinary_grad",
    "6_qnac_fixed_shots", "7_sf_qnac_ids",
]
SHORT_LABELS = ["XGBoost", "SHO-\nXGBoost", "QRL-SHO-\nXGBoost\n(repro.)",
                "Classical\nAC", "VQC-AC\n(ord. grad)", "QNAC\n(fixed shots)", "SF-QNAC-\nIDS"]


def _box(ax, xy, w, h, text, fc, ec="#333333", fontsize=9):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                          linewidth=1.1, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, wrap=True)


def _arrow(ax, p1, p2, color="#333333"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                  linewidth=1.2, color=color))


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    boxes = [
        (0.2, 2.0, 1.5, 1.0, "IDS Dataset\n(leakage-safe\nsplit)", "#EAEDF1"),
        (2.1, 2.0, 1.6, 1.0, "RL Optimization\nEnvironment\n(compact state)", "#DCEFEF"),
        (4.1, 3.1, 1.7, 1.0, "4-qubit VQC\nActor\n(data re-upload)", C_PROP),
        (4.1, 0.8, 1.7, 1.0, "Classical\nCritic (MLP)", "#EAEDF1"),
        (6.2, 2.0, 1.8, 1.0, "Sparse feature\ngroups + XGBoost\nhyperparameters", "#EAEDF1"),
        (8.4, 2.0, 1.4, 1.0, "Final XGBoost\nClassifier", "#DCEFEF"),
    ]
    for x, y, w, h, t, c in boxes:
        _box(ax, (x, y), w, h, t, c)
    _arrow(ax, (1.7, 2.5), (2.1, 2.5))
    _arrow(ax, (3.7, 2.7), (4.1, 3.4))
    _arrow(ax, (3.7, 2.3), (4.1, 1.3))
    _arrow(ax, (5.8, 3.5), (6.2, 2.7))
    _arrow(ax, (5.8, 1.3), (6.2, 2.3))
    _arrow(ax, (8.0, 2.5), (8.4, 2.5))
    ax.annotate("action (feature-group toggle /\nhyperparameter nudge)", xy=(6.0, 2.9), fontsize=7.5,
                ha="center", color="#555")
    ax.annotate("validation reward\n(task + quantum-cost terms)", xy=(3.9, 0.35), fontsize=7.5, ha="center", color="#555")
    ax.set_title("SF-QNAC-IDS: quantum computation confined to the compact\noptimization-state RL loop, not per-sample feature encoding", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_architecture.png", bbox_inches="tight")
    plt.close(fig)


def fig2_flow_comparison():
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4))
    for ax, title, boxes, note in [
        (axes[0], "Reference: QRL-SHO-XGBoost", [
            "Every\nnetwork\nflow", "8-qubit\nquantum\nembedding\n(per sample)",
            "Wrapper SHO\n(N horses x T\niters, full\nretrain each)", "Tabular\nQ-learning", "XGBoost"],
         "Quantum processing repeated for EVERY sample, every search candidate"),
        (axes[1], "Proposed: SF-QNAC-IDS", [
            "Compact RL\noptimization\nstate", "4-qubit VQC\nactor\n(state only)",
            "Surrogate-free\ndirect policy\nsearch", "Classical\ncritic + QNPG", "XGBoost"],
         "Quantum processing touches only the small optimization state"),
    ]:
        ax.set_xlim(0, 10); ax.set_ylim(0, 2.2); ax.axis("off")
        ax.set_title(title, fontsize=10.5, loc="left")
        n = len(boxes)
        w = 1.6
        xs = np.linspace(0.2, 10 - w - 0.2, n)
        for i, (x, t) in enumerate(zip(xs, boxes)):
            fc = C_PROP if ("quantum" in t.lower() or "qubit" in t.lower()) else "#EAEDF1"
            _box(ax, (x, 0.7), w, 1.0, t, fc, fontsize=8.3)
            if i < n - 1:
                _arrow(ax, (x + w, 1.2), (xs[i + 1], 1.2))
        ax.annotate(note, xy=(5, 0.25), fontsize=8.5, ha="center", color="#444")
    fig.suptitle("Where quantum computation happens: per-sample (reference) vs.\noptimization-state-only (proposed)", fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_flow_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _load_results():
    p = RESULTS_DIR / "raw_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df[df["status"] == "ok"].copy()


def fig3_f1_vs_shots(df):
    if df is None or "total_shots" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    g = df.groupby("method").agg(f1=("test_f1_macro", "mean"), shots=("total_shots", "mean")).reset_index()
    g = g[g["shots"] >= 0]
    for _, row in g.iterrows():
        is_prop = row["method"] == "7_sf_qnac_ids"
        is_ref = row["method"] == "3_qrl_sho_xgboost_reproduction"
        color = C_PROP if is_prop else (C_MID if is_ref else C_BASE)
        marker = "*" if is_prop else ("D" if is_ref else "o")
        size = 220 if is_prop else (120 if is_ref else 70)
        ax.scatter(max(row["shots"], 1), row["f1"] * 100, color=color, marker=marker, s=size,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(row["method"].replace("_", " "), (max(row["shots"], 1), row["f1"] * 100),
                    fontsize=6.6, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("symlog")
    ax.set_xlabel("Total quantum measurement shots (mean across configs, symlog)")
    ax.set_ylabel("Test Macro-F1 (%)")
    ax.set_title("Macro-F1 vs. total quantum shots")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_f1_vs_shots.png", bbox_inches="tight")
    plt.close(fig)


def fig4_f1_vs_time(df):
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    g = df.groupby("method").agg(f1=("test_f1_macro", "mean"), t=("wall_time_total_sec", "mean")).reset_index()
    for _, row in g.iterrows():
        is_prop = row["method"] == "7_sf_qnac_ids"
        is_ref = row["method"] == "3_qrl_sho_xgboost_reproduction"
        color = C_PROP if is_prop else (C_MID if is_ref else C_BASE)
        marker = "*" if is_prop else ("D" if is_ref else "o")
        size = 220 if is_prop else (120 if is_ref else 70)
        ax.scatter(max(row["t"], 0.01), row["f1"] * 100, color=color, marker=marker, s=size,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(row["method"].replace("_", " "), (max(row["t"], 0.01), row["f1"] * 100),
                    fontsize=6.6, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Mean wall-clock time per run (s, log scale)")
    ax.set_ylabel("Test Macro-F1 (%)")
    ax.set_title("Macro-F1 vs. training/optimization wall time")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_f1_vs_time.png", bbox_inches="tight")
    plt.close(fig)


def fig5_ablation_bars(df):
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ab = df[df["method"].isin(ABLATION_ORDER)]
    ds_protos = sorted(ab.apply(lambda r: f"{r['dataset']}-{r['protocol']}", axis=1).unique())
    x = np.arange(len(ABLATION_ORDER))
    width = 0.8 / max(len(ds_protos), 1)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(ds_protos)))
    for i, dsp in enumerate(ds_protos):
        ds, proto = dsp.rsplit("-", 1)
        sub = ab[(ab["dataset"] == ds) & (ab["protocol"] == proto)]
        means = [sub[sub["method"] == m]["test_f1_macro"].mean() * 100 for m in ABLATION_ORDER]
        ax.bar(x + i * width - 0.4 + width / 2, means, width, label=dsp, color=cmap[i])
    ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=7.8)
    ax.set_ylabel("Test Macro-F1 (%)")
    ax.set_title("Ablation ladder: where does any improvement come from?")
    ax.legend(fontsize=7.5, ncol=len(ds_protos))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_ablation_bars.png", bbox_inches="tight")
    plt.close(fig)


def main():
    fig1_architecture()
    fig2_flow_comparison()
    df = _load_results()
    fig3_f1_vs_shots(df)
    fig4_f1_vs_time(df)
    fig5_ablation_bars(df)
    print("Figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
