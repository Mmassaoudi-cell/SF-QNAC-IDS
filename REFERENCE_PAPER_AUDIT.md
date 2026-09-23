# REFERENCE_PAPER_AUDIT.md

**Audited paper:** S. Kumar, M. Dwivedi, M. Kumar, H. Wu, S. S. Gill, "Explainable AI-Enabled
Intrusion Detection in IoT Networks Using Optimized Quantum Reinforcement Learning,"
*IEEE Transactions on Consumer Electronics*, 2026, DOI 10.1109/TCE.2026.3715996.

Source: full 12-page PDF read directly (all pages) earlier in this session. This document
records exactly what is and is not reproducible from the paper's own text, so later reproduction
code cannot silently invent unstated parameters.

---

## 1. What the paper states explicitly (reproducible)

| Item | Value stated in paper | Location |
|---|---|---|
| Qubit count | 8 | §III.A–D, Table VI, Alg. 1 |
| Encoding | Angle encoding, 1 raw feature → 1 qubit via `R_Y(θ)`, θ = π·(x−min)/(max−min) | Eq. 17, Alg. 1 lines 19–22 |
| Entanglement | Ring topology: `CNOT(i,i+1)` for i=1..7, plus closing `CNOT(8,1)` | Eq. 13, 18, Alg. 1 lines 24–28 |
| Variational layer | Per qubit: `R_Z(φ⁽³⁾)·R_Y(φ⁽²⁾)·R_X(φ⁽¹⁾)` | Eq. 14, Alg. 1 lines 30–32 |
| Measurement | Pauli-Z expectation per qubit, ⟨ψ\|Z_q\|ψ⟩ | Eq. 5, 15, 16, Alg. 1 lines 34–35 |
| Shots | 8192 per circuit execution | §III.D text, p.4 |
| Noise model | Depolarizing + measurement-error mitigation + Zero-Noise Extrapolation (ZNE/"mitiq") | §III.D text, p.4 |
| Backend | Qiskit Aer `aer_simulator` (no real hardware) | §IV intro |
| Feature selection (one-time) | `FEATURES()`: train ONE preliminary XGBoost, rank features by fraction of trees using each feature (Eq. 28), take top-k | Algorithm 2, lines 3–6 |
| k (selected features) | 8 (fixed, matches qubit count) | Algorithm 2 line 7 |
| Hyperparameter/feature search loop | `RL-SHO`: N "horse" candidates, T_max iterations, **each horse fully retrains XGBoost every iteration** | Algorithm 2, lines 8–22 |
| RL formulation | Tabular Q-learning, Q(s,a) update Eq. 20/29, state = {F1_val, \|S\|, θ}, ε-greedy | §III.B, Eq. 20, 23 |
| RL hyperparameters | α=0.10, γ=0.95, ε: 1.0→0.01 decay 0.995, 5 discrete actions, 1500 episodes | Eq. 23, §III.B text |
| Classifier | XGBoost, standard regularized objective (Eq. 24–25) | §III.C |
| Preprocessing | Median imputation, min-max scaling to [0,1], one-hot/label encoding of categoricals, class-weighted XGBoost | §IV.B text |
| Split | Stratified 70% train / 15% val / 15% test | §IV.B text, §IV.D text |
| Cross-dataset protocol | Stratified 5-fold CV mentioned once in §IV.D text alongside the 70/15/15 split — **internally inconsistent, see §3 below** | §IV.D |
| Datasets | UNSW-NB15 (2,540,044 rows, 49 features, 9 attack categories); CIC-DDoS2019 (12,452,505 rows, 88 features); Application-Layer DDoS (3,085,728 rows, 21 features, Kaggle `wardac/applicationlayer-ddos-dataset`) | §IV.A |
| Headline results | 99.20% / 99.21% / 99.53% test accuracy on UNSW-NB15 / CIC-DDoS2019 / App-DDoS respectively; F1 > 99.19% all three | Abstract, Table II |
| Quantum circuit scalability (Table VI) | At 8 qubits: depth 24 (post-transpilation), 72 1-qubit gates, 24 2-qubit gates, fidelity 0.9815–0.9954, runtime 9.85–10.15 simulator-hours | Table VI |
| Explainability | LIME (`LimeTabularExplainer`) on final XGBoost predictions, over original (not qubit) feature space | §IV.H, Fig. 5 |
| Ablation | Table VIII: removes quantum encoding, reduces state, removes reward shaping, removes adaptation, uses shallow network, removes exploration — reports accuracy/F1/return deltas | §IV.E |

## 2. What the paper reports but does NOT specify precisely enough to reproduce exactly

| Missing parameter | Why it matters | What we will have to assume/document at reproduction time |
|---|---|---|
| SHO population size N and iteration count T_max | Directly determines the paper's own claimed "wrapper" cost | Not given anywhere as a number. Must be chosen and stated as an assumption in any reproduction, not inferred from the paper. |
| Variational ansatz parameters (φ) | Whether trained, fixed-random, or data-re-uploaded across layers is never stated | Paper never describes a training rule for the "quantum embedding" weights separately from the RL-tuned XGBoost hyperparameters. Any reproduction that fixes these (e.g., a seeded random ansatz) is an assumption, not a stated fact. |
| Relationship between §III.A's generic SHO position-update (Eq. 1–3, binary feature mask via sigmoid threshold) and Algorithm 2's `RL-SHO` (per-horse XGBoost retrain, θ described as "hyperparameters") | The paper blends feature-mask search and hyperparameter search into language that reads as one loop in places and two in others | This is the single most consequential ambiguity for reproducing "wrapper cost." Must be stated explicitly as a modeling choice in any reproduction (e.g., treat the horse vector as jointly encoding a feature mask AND hyperparameters), not presented as verified paper behavior. |
| Reward weight λ in `R = F1_val − λ·(|S|/d)` | Governs sparsity pressure | No numeric value given. |
| XGBoost hyperparameter search ranges (n_estimators, max_depth, etc. bounds) | Needed to bound the RL/SHO action space | Not specified. |
| "Five discrete actions" — their exact semantics | Needed to reproduce the action space | Paper states there are five, does not enumerate what each does. |
| Number of independent runs behind Table III's "mean of independent runs" | Needed for a variance-matched reproduction | Not stated as a count. |
| §IV.D's own protocol: is it stratified 70/15/15 (as stated in §IV.B and reiterated at the top of §IV.D) or stratified 5-fold CV (as stated later in the same paragraph of §IV.D)? | Affects how "cross-dataset validation" numbers in Table VII should be reproduced | Both are stated in the same section without reconciling which applies where. Any reproduction must pick one and say so. |
| Exact preliminary-XGBoost hyperparameters used inside the one-time `FEATURES()` ranking step | Downstream feature selection depends on it | Not specified. |

## 3. Internal inconsistencies worth flagging (not just gaps)

1. **SHO vs. RL-SHO scope.** §III.A presents Sea-Horse Optimization as a generic *binary feature-selection* algorithm (Eq. 1–3). §III.D and Algorithm 2 present "RL-SHO" as a *hyperparameter* search where each of N "horses" is a hyperparameter vector θ, fully retrained every iteration. The paper's own running text in §III.D ("RL-SHO selects informative features by optimizing feature subset iteratively," Eq. 32) then re-attributes feature selection to this same loop, contradicting Algorithm 2's separate one-time `FEATURES()` procedure. A faithful reproduction cannot resolve this from the text alone; it must state which interpretation it implements.
2. **§IV.D protocol conflict** (stratified 70/15/15 vs. stratified 5-fold), noted above.
3. **Table VI vs. the qubit-encoding equations.** Table VI's reported 8-qubit circuit (depth 24, 72 1-qubit / 24 2-qubit gates) is roughly 2–3× the gate count implied by directly executing Eq. 11–14 once (encoding + ring entanglement + one variational layer ≈ 8 RY + 8 CNOT + 24 variational rotations = 32 1-qubit / 8 2-qubit gates at the logical-circuit level). The 2–3× inflation is consistent with Zero-Noise-Extrapolation gate folding (which the paper says it applies, §III.D) being included in Table VI's post-transpilation count, but the paper never states this explicitly — it is our inference, not a paper claim, and must be labeled as such wherever cited.
4. **Runtime attribution.** The paper attributes "9.85-hour simulator runtime" (8 qubits) to "repetitive circuit sampling, reinforcement learning optimization, and error-mitigation processes" collectively (§III.D text) without breaking down how much is attributable to shots vs. episode count vs. SHO wrapper retrains vs. ZNE. Any efficiency comparison against this number must treat it as a single bulk figure, not decompose it as if the paper had reported component-wise timings.

## 4. What this means for any reproduction

- A "faithful reproduction" of QRL-SHO-XGBoost is **necessarily a reconstruction**, not a literal re-implementation, for every item in §2–3 above. Any reproduction code must document each assumed value inline (population size, iteration count, ansatz training rule, reward λ, action semantics, split-protocol choice) and must not present its own reproduced numbers as equivalent to the paper's reported numbers.
- The paper's own headline accuracies (99.20–99.53%) were obtained on the **full, non-subsampled** datasets (2.54M / 12.45M / 3.09M rows) with the paper's unstated-but-presumably-large N/T_max and full 1500-episode RL budget. Any reproduction run at reduced scale (subsampled rows, smaller search budgets) must report its own numbers as pilot/reduced-scale and must not be compared directly to the paper's headline percentages as if on equal footing.
