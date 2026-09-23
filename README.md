# SF-QNAC-IDS — complete (Phase 1 + Phase 2)

Shot-Frugal Quantum Natural Actor-Critic Optimization for efficient intrusion detection,
developed against the reference paper "Explainable AI-Enabled Intrusion Detection in IoT
Networks Using Optimized Quantum Reinforcement Learning" (Kumar et al., IEEE TCE, 2026).

**Scope note:** this began as Phase 1 of a staged research plan, scoped down from a much
larger 40-section specification by an explicit user decision (see the top-level conversation)
to keep a single session's compute budget realistic. Phase 2 then ran every item Phase 1 had
explicitly deferred: 7-seed statistics, shot/qubit/depth sensitivity grids, NISQ-like noise
robustness, multiclass evaluation, and cross-dataset transfer. Exact scope and every number's
provenance is in `FINAL_RESEARCH_SUMMARY.md`.

## Directory layout

```
REFERENCE_PAPER_AUDIT.md    what the reference paper does/doesn't specify, and every
                             assumption made to reproduce it anyway
DATASET_SELECTION.md        inventory of C:\Users\MMASSAOUDI\Desktop\Data and why these
                             3 datasets were selected
FINAL_RESEARCH_SUMMARY.md   headline numbers, provenance, and remaining limitations
README.md                   this file
code/                       all source
  quantum_sim.py            exact NumPy statevector simulator + batched embedding
  noise_sim.py               Phase 2: Monte-Carlo depolarizing + analytic readout noise,
                             mitigation (readout inversion, ZNE)
  data_pipeline.py          Protocol A/B loaders, deduplication, leakage audit, disk cache
                             for large source files (cache/*.parquet)
  baseline_repro.py         QRL-SHO-XGBoost reference reproduction (ablation rung 3)
  qnac_core.py               SF-QNAC-IDS core: env, quantum actor, classical critic,
                             QNPG, adaptive shots, cost-aware reward (ablation rungs 4-7)
  ablation_ladder.py         unifies all 7 ablation rungs behind one call signature
  benchmarks.py               6-method benchmark suite (LogReg/RF/LightGBM/Optuna-XGBoost/
                             quantum-kernel classifier, + XGBoost-default = rung 1)
  run_all.py                  orchestrates the full grid (now 7 seeds), writes
                             results/raw_results.jsonl (schema-safe) and rebuilds
                             results/raw_results.csv
  stats_analysis.py           paired significance testing on the headline comparison
  make_tables.py              CSV + LaTeX tables from raw_results.csv (Phase 1)
  make_figures.py             figures from raw_results.csv (Phase 1) + 2 schematic diagrams
  shot_sensitivity.py         Phase 2: 64-8192 shot sweep on the QNAC config
  qubit_depth_sensitivity.py  Phase 2: 2/4/6/8 qubits x 1/2/3/4 layers grid
  noise_experiment.py         Phase 2: NISQ-like noise-robustness eval of the trained policy
  multiclass_eval.py          Phase 2: real multiclass labels, macro metrics, per-class report
  cross_dataset_transfer.py   Phase 2: reconciled 3-feature transfer, zero/few-shot
  make_phase2_outputs.py      CSV + LaTeX tables + figures for all 5 Phase-2 studies
  smoke_test_all.py           tiny-scale correctness check (run before the full grid)
cache/                      cached parquet pools for large source files (built once, reused
                             across seeds -- see data_pipeline.py's _cached_pool)
splits/                     per-run split manifests (sizes, duplicate audit, positive rate)
results/                    raw_results.jsonl / .csv, noise_robustness.csv,
                             shot_sensitivity.csv, qubit_depth_sensitivity.csv,
                             multiclass_results.json, cross_dataset_transfer.csv, run logs
tables/                     generated CSV + .tex tables (Phase 1 + Phase 2)
figures/                    generated PNG figures (fig1-9)
manuscript/                 main.tex (IEEEtran, 11 pages) -- pulls tables/figures by \input /
                             \includegraphics, no hand-typed result numbers; 8 verified
                             citations in Related Work
```

## Reproducing the full run

```bash
cd code
python data_pipeline.py        # sanity-check all loaders + save manifests
python smoke_test_all.py       # tiny-scale correctness check (~15s)
python run_all.py              # full grid: 3 datasets x protocols x 7 seeds x 12 methods
                                # (resumable -- re-running skips completed rows, matched on
                                # (dataset, protocol, seed, method) from the JSONL log; in
                                # this environment individual background runs were capped at
                                # ~10 min, so run_all.py was invoked multiple times in
                                # sequence and picked up where it left off each time)
python stats_analysis.py       # writes tables/stats_headline_comparison.csv
python make_tables.py          # writes tables/*.csv and tables/*.tex (Phase 1)
python make_figures.py         # writes figures/fig1-5 (Phase 1)

# Phase 2 (each independent; run in any order)
python shot_sensitivity.py
python qubit_depth_sensitivity.py
python noise_experiment.py
python multiclass_eval.py
python cross_dataset_transfer.py
python make_phase2_outputs.py  # writes tables + figures fig6-9 for all 5 Phase-2 studies
```

To compile the manuscript (MiKTeX/pdflatex confirmed available in this environment):
```bash
cd manuscript
pdflatex main.tex
pdflatex main.tex   # twice more, for references/table/figure numbering
pdflatex main.tex
```

## Environment

Python 3.13, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `scipy`,
`matplotlib`, `pyarrow` (for the parquet cache). No external quantum SDK (Qiskit, PennyLane,
etc.) is used or required -- `quantum_sim.py` and `noise_sim.py` are from-scratch exact/
Monte-Carlo simulators, sufficient for the $\le 8$-qubit circuits used throughout. See
`requirements.txt`.

## What every number traces back to

Every number in `manuscript/main.tex`'s tables comes from `\input`-ing a `.tex` file
generated by `make_tables.py` or `make_phase2_outputs.py` directly from a `results/*.csv` or
`.json` file; every figure is `\includegraphics`-ed from a PNG generated the same way. No
result number is hand-typed into the manuscript. `results/raw_results.jsonl` is the
append-only, schema-safe log the main grid writes to; `raw_results.csv` is rebuilt from it
after every dataset/protocol/seed block completes. The 5 Phase-2 studies each write their own
dedicated CSV/JSON directly (no intermediate JSONL needed, since each is a single bounded
script run rather than a long resumable grid).
