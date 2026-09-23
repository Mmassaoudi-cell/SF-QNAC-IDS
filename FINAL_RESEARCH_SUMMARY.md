# FINAL_RESEARCH_SUMMARY.md — Complete (Phase 1 + Phase 2)

STATUS: **complete**. Phase 1's full experiment grid (180 runs) was extended to 7 seeds
(420 runs total, 0 errors), and every item Phase 1 explicitly deferred has been run: shot-
sensitivity sweep, qubit×depth sensitivity grid, NISQ-like noise-robustness study, multiclass
evaluation, and cross-dataset transfer. Every number below traces to a file under `results/`
or `tables/`; none is hand-typed without that provenance. The manuscript
(`manuscript/main.pdf`) compiles cleanly to 11 pages with zero unresolved references, and
its Related Work section cites 8 sources, each located and verified via live web search this
phase (titles/authors/venues cross-checked against a publisher page, arXiv listing, or
indexing service before being entered — none asserted from memory).

## Scope executed

**Phase 1 (core program):** 3 datasets (UNSW-NB15, CIC-IDS-2017, CICIoT23), Protocol A
(paper-compatible) for all 3 + Protocol B (real leakage-safe disjoint split) for CIC-IDS-2017
and CICIoT23, a 7-rung ablation ladder, a 6-method benchmark suite, a duplicate/leakage audit.

**Phase 2 (this update):**
- Seeds extended from 3 to **7** (`{42..48}`) — the upper end of the 5–10 range targeted.
- Shot-sensitivity sweep: 64–8192 shots, 2 datasets × 2 seeds (`shot_sensitivity.py`).
- Qubit×depth sensitivity grid: 2/4/6/8 qubits × 1/2/3/4 layers, 2 datasets, 1 seed
  (`qubit_depth_sensitivity.py`).
- NISQ-like noise-robustness study: Monte-Carlo depolarizing + analytic readout noise, with
  readout mitigation and ZNE, applied only at final-policy-evaluation time, 3 datasets × 2
  seeds (`noise_sim.py`, `noise_experiment.py`).
- Multiclass evaluation: real attack-type labels (not binarized), all 3 datasets, reusing
  SF-QNAC-IDS's binary-search-derived feature/hyperparameter choice with a multi:softprob
  objective (`multiclass_eval.py`).
- Cross-dataset transfer: a reconciled 3-feature common representation (duration, packet
  rate, total bytes), in-domain / zero-shot / few-shot (n=300) across all 6 ordered dataset
  pairs (`cross_dataset_transfer.py`).
- Literature review: 8 verified citations added to the manuscript's Related Work.

**Still out of scope, stated not hidden:** full-scale (non-subsampled, multi-million-row)
validation; a discretization-insensitive convergence criterion (the episodes-to-convergence
anomaly is flagged, not fixed); a multiclass-aware search objective (multiclass eval reuses a
binary-derived configuration); device-calibrated (vs. illustrative) noise modeling; a
verified-unit-conversion cross-dataset representation beyond the 3-feature minimal one; full
SHAP/quantum-policy-attribution explainability outputs; a full systematic literature review
(Related Work cites verified, directly relevant sources, not an exhaustive survey).

## Headline numbers (7-seed, final)

| Item | Value | Source |
|---|---|---|
| Qubits | 4 (SF-QNAC-IDS) vs. 8 (reference) | fixed by design |
| Circuit depth / gates | SF-QNAC-IDS: depth 11, 24 1-qubit / 6 2-qubit gates; reference: depth 12, 32/8 | `tables/quantum_efficiency.csv` |
| Shot strategy | adaptive 128–4096 (policy margin/entropy/phase); reference fixed 8192 | `qnac_core.py::adaptive_shots` |
| Mean test Macro-F1 (reference → SF-QNAC-IDS, 5 cells) | CIC-IDS-2017-A: 86.26%→99.15% (+12.9pp); CIC-IDS-2017-B: 60.81%→62.68% (+1.9pp, **not significant**); CICIoT23-A: 80.50%→92.25% (+11.7pp); CICIoT23-B: 82.82%→92.23% (+9.4pp); UNSW-NB15-A: 85.06%→90.18% (+5.1pp) | `tables/stats_headline_comparison.csv` |
| Statistical significance (7 seeds) | Holm-corrected paired-t: CIC-A $p=1.6\times10^{-4}$, CICIoT-A $p=1.9\times10^{-3}$, CICIoT-B $p=7.1\times10^{-4}$, UNSW $p=4.0\times10^{-5}$ (all significant); CIC-IDS-2017-B $p=0.64$ (**not significant, CI spans zero**); Wilcoxon reaches its n=7 minimum ($p=0.0156$) on 4/5 cells | `tables/stats_headline_comparison.csv` |
| Full-cost classifier retrains | reference: 142 (all full-cost); SF-QNAC-IDS: 61 total = 51 low-fidelity + **9 full-fidelity** → **15.8× fewer expensive retrains** | `results/raw_results.csv` |
| Total quantum shots (pooled mean) | reference: 2,141,647,xxx (~2.14B); SF-QNAC-IDS: 153,600 → **~13,943× fewer** | `tables/quantum_efficiency.csv` |
| Shot reduction from adaptive allocation alone | QNAC (fixed 8192): 1,966,080 shots vs. SF-QNAC-IDS (adaptive): 153,600 → **12.8×**, isolated from the natural-gradient update | `tables/ablation_ladder.csv` |
| Mean wall-clock time | reference: 55.23s; SF-QNAC-IDS: 7.54s → **~7.3× faster** | `tables/quantum_efficiency.csv` |
| Strongest overall benchmark | **Random Forest**, not SF-QNAC-IDS: 88.8% mean Macro-F1 (vs. SF-QNAC-IDS 87.3%) at ~9× less training time (0.88s vs. 7.54s) | `tables/main_results.csv` |
| UNSW-NB15 exact-duplicate rate (pre-dedup) | **40.3%** (103,989/257,673 rows) | `splits/*manifest.json` |
| Shot-sensitivity finding | Test Macro-F1 statistically flat 64→8192 shots (91.1%→91.8%, within noise); policy KL-divergence from ideal shrinks monotonically 2+ orders of magnitude over the same range | `results/shot_sensitivity.csv` |
| Qubit/depth-sensitivity finding | Best cell (93.7% F1) tied between 2-qubit/2-layer and 8-qubit/2-layer configs; worst (89.5%) is 4-qubit/1-layer — more qubits/depth does not predict higher accuracy | `results/qubit_depth_sensitivity.csv` |
| Noise-robustness finding | Readout noise: negligible effect (mean KL≈5e-5), exactly recoverable by mitigation; depolarizing noise: mean KL≈3.5e-3 (~70× larger), flips argmax action in 50% of tested cells; ZNE reduces KL by ~2.4× but doesn't eliminate all action flips; downstream accuracy impact of flips was small (<2pp) in this bounded check | `results/noise_robustness.csv` |
| Multiclass finding | Accuracy vs. balanced accuracy gap on all 3 datasets: UNSW-NB15 83.1%/52.0%; CIC-IDS-2017 99.7%/82.2%; CICIoT23 98.7%/78.0%. UNSW-NB15's rarest classes (Backdoor n=6, Worms n=3 in test fold) score **0.0 F1** | `results/multiclass_results.json` |
| Cross-dataset transfer finding | Zero-shot Macro-F1 as low as 0.03 (CIC-IDS-2017→CICIoT23), never exceeds 0.49 across 6 pairs; few-shot (n=300) recovers to 0.74–0.79 in every pair | `results/cross_dataset_transfer.csv` |

## Strongest benchmark — reported honestly, not minimized

Unchanged from Phase 1's finding, now confirmed at 7-seed scale: **Random Forest has the
highest mean Macro-F1 of any method tested (quantum or classical) across all 5 cells**,
beating SF-QNAC-IDS by 1.5 points at ~9× less training time. SF-QNAC-IDS's real, defensible
advantage is specifically against the **quantum** reference method (QRL-SHO-XGBoost) — both
more accurate (4/5 cells, now at high statistical confidence) and far cheaper — not against
strong classical baselines in general.

## Ablation attribution

- Accuracy gain over the reference is attributable to leaving wrapper-SHO/per-sample quantum
  encoding behind (rung 3→4: 79.1%→86.6%), not to which specific actor-critic variant is used
  on top (rungs 4–7 span only 1.2 points; QNAC's fixed-shot accuracy is marginally *above*
  full SF-QNAC-IDS).
- Shot-count reduction is attributable specifically to adaptive allocation (12.8×, isolated
  from the natural-gradient update).
- Episodes-to-convergence is *higher* for every actor-critic rung than for the tabular
  reference — contrary to this proposal's own hypothesis, persists at 7 seeds, flagged as a
  likely discretization artifact (Sec. VI-C of the manuscript), not fixed this phase.

## Remaining limitations (stated, not hidden)

See manuscript Sec. VI-K (`Limitations`) for the full 11-point list, condensed here:
1. Still pilot-scale (20,000-row working subsamples), not the reference paper's full-dataset
   regime — Phase 2 extended seeds and study breadth, not sample size.
2. Reference reproduction's SHO/hyperparameter-search interpretation is a stated modeling
   choice (`REFERENCE_PAPER_AUDIT.md` Sec. 3.1), not verified against the paper's actual code.
3. Feature groups are contiguous raw-column chunks, not semantically curated.
4. Natural-gradient update uses a diagonal *empirical* Fisher (via exact parameter-shift), not
   the literal quantum Fisher information matrix.
5. Noise model uses illustrative, not device-calibrated, error rates, bounded to 3 datasets ×
   2 seeds.
6. 7 seeds resolves 4/5 cells to significance; CIC-IDS-2017-B's non-significance is now
   well-supported (not just under-tested), but 10 seeds was not reached.
7. Multiclass evaluation reuses a binary-derived configuration; several rarest source classes
   did not survive working-subsample stratification (documented per-dataset, not hidden).
8. Cross-dataset transfer's 3-feature representation is reconciled by name/semantics, not by
   verified unit conversion against each dataset's original documentation.
9. Episodes-to-convergence artifact remains unresolved.
10. No full systematic literature review (verified, directly relevant citations only).
11. No full SHAP/quantum-policy explainability outputs run this phase.

## Reproducibility

Every number traces to a file under `results/` or `tables/`, generated by scripts under
`code/`, with fixed seeds and every hyperparameter stated inline in the corresponding script.
See `README.md` for exact reproduction commands (now including all Phase 2 scripts). The
manuscript PDF is `manuscript/main.pdf` (11 pages, MiKTeX/pdflatex, zero unresolved
references, zero fabricated citations).
