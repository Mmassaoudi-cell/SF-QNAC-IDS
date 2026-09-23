# Dataset inventory and selection

Inventory performed over `C:\Users\MMASSAOUDI\Desktop\Data` (all top-level folders listed and
sampled). Full per-file sizes/headers checked before selection; selection criteria were fixed
*before* looking at any candidate method's results on these datasets.

## Availability against the reference paper's three datasets

| Paper dataset | Locally available? | Notes |
|---|---|---|
| UNSW-NB15 | **Yes** — `UNSW_NB15/UNSW_NB15_{training,testing}-set.csv` | 257,673 rows combined, 49 columns (`id`, 42 features, `attack_cat`, `label`). Matches the paper's dataset identity, not just its name. |
| CIC-DDoS2019 | **No** | Not present anywhere under `Desktop\Data`. Not fabricated or substituted silently — see replacement below. |
| Application-Layer DDoS (Kaggle `wardac/applicationlayer-ddos-dataset`, 3,085,728 rows / 21 features per the paper) | **No** | `APA-DDoS-Dataset/APA-DDoS-Dataset.csv` exists (151,201 rows, 23 columns: `ip.src`, `ip.dst`, TCP/IP flags, frame stats, `Label`) but is a **different dataset** — different size, different feature set, different source. It is *not* used as a stand-in for the paper's Application-Layer DDoS dataset in any comparison, to avoid misrepresenting a reproduction as exact when it isn't. |

## Replacement selection (locality + criteria, not result-shopping)

Selection criteria, fixed in advance: (1) data quality / cleanliness, (2) representativeness of
modern IDS/IoT traffic, (3) feasibility of a genuine leakage-safe (Protocol B) split, (4) multiclass
label availability for future phases, (5) scale relative to available compute budget for this
phase, (6) not already saturated/trivial for tree ensembles.

| Dataset | Rows (raw) | Features | Multiclass labels | Protocol B feasibility | Decision |
|---|---:|---:|---|---|---|
| **CIC-IDS-2017** | ~2.83M across 8 day-based capture files | 78 (+ Label) | Yes — per-file attack types (DDoS, PortScan, Web Attack, Infiltration, Botnet, Brute-Force) | **Strong** — Monday is 100% benign, other days carry specific attack types; a genuine day/capture-disjoint split is possible without any invented heuristic | **Selected.** Best available local substitute for CIC-DDoS2019: same CIC lineage, modern flow-based features, and uniquely offers a real disjoint-capture split. |
| **CICIoT23** | 5,491,971 (train file alone) | 46 (+ label) | Yes — 34 fine-grained attack labels (DDoS/DoS/Mirai/Recon/Spoofing/MITM/etc.) | **Strongest** — ships with an **official** pre-made train/test/validation split; using it directly is the cleanest possible Protocol B, no custom disjointness heuristic needed | **Selected.** Modern (2023), large-scale, heavily imbalanced (benign = 2.4% of train) — a genuine difficulty/minority-recall stress test, and the one dataset in this selection with a truly official split. |
| Edge-IIoTset (`ML-EdgeIIoT-dataset.csv`) | large (82 MB curated file) | many, curated | Yes | Plausible via its Attack_type/Attack_label fields | Good candidate; **deferred to Phase 2** to keep this phase at the agreed 3-dataset cap. |
| RT-IoT2022 | 123,117 | 83 | Yes (`Attack_type`) | Weak (no native session/time-disjoint field) | Already used in a prior pilot this session (see earlier Squire-IDS work); **deferred to Phase 2** as a 4th dataset rather than included here, to keep this phase's compute budget bounded. |
| Bot_IoT | 74 files, ~200 MB each (~15 GB total) | many | Yes | Plausible (per-file time ordering) | **Excluded from this phase** — total volume far exceeds this phase's compute budget; a stratified single-file subsample would misrepresent the dataset's actual scale. |
| KDD (`kdd_train.csv`/`kdd_test.csv`) | moderate | 41 | Yes | Has an official split | **Excluded** — 1999-vintage traffic characteristics are a poor fit for an IoT-focused paper's central claims; kept as a possible robustness/legacy-comparison dataset for a later phase, not a headline dataset. |
| APA-DDoS-Dataset | 151,201 | 23 | Binary only (`Label`) | Weak | **Excluded** as a headline dataset (see note above on not conflating it with the paper's App-DDoS dataset); could serve Phase 2 as an extra binary DDoS check. |

## Final selection for this phase

1. **UNSW-NB15** — direct identity match with the reference paper; used for the paper-compatible
   Protocol A comparison.
2. **CIC-IDS-2017** — modern local substitute for the unavailable CIC-DDoS2019; provides this
   phase's genuine day-disjoint Protocol B split.
3. **CICIoT23** — modern, large-scale, heavily imbalanced IoT dataset with an official train/test/
   validation split, used as this phase's second (and cleanest) Protocol B evaluation.

All three are used at a **20,000-row stratified working subsample** per dataset for this phase
(documented per-run in each result CSV's `n_samples` column), to keep the full ablation ladder ×
benchmark suite × 3-seed grid within a single session's realistic compute budget. This is a stated
scope reduction, not a hidden one — full-scale runs are a defined Phase 2 item.
