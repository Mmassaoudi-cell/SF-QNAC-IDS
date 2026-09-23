"""
Phase 2: NISQ-like noise robustness evaluation of the FINAL trained
SF-QNAC-IDS policy (not injected during every training step -- consistent
with the design brief's guidance to reserve expensive noise/mitigation
handling for final refinement/evaluation, which is itself part of this
method's shot-frugal efficiency story).

Trains SF-QNAC-IDS once per (dataset, protocol, seed) cell in a small
representative subset (bounded for compute reasons -- 3 cells, 2 seeds each
= 6 trainings), then evaluates the trained actor's policy at its own final
state under 5 conditions:
  1. ideal            -- exact statevector (0 shots)
  2. finite-shot       -- 1024-shot sampled read-out, no gate/readout noise
  3. depolarizing      -- Monte-Carlo depolarizing noise (see noise_sim.py)
  4. readout           -- symmetric bit-flip readout noise only
  5. combined          -- depolarizing + readout together
plus mitigation variants (readout-error mitigation on 'readout'/'combined';
ZNE on 'combined') applied ONLY at this final-evaluation stage.

For every condition, records: policy KL-divergence from the ideal policy,
whether the noisy policy's argmax action differs from the ideal one, and --
if it differs -- re-plays the environment with the noise-selected action and
does a full downstream XGBoost fit/eval, so "does noise change the decision"
and "does that changed decision hurt IDS accuracy" are both measured, not
just the first one.

Noise parameters (p_1q=0.01, p_2q=0.03, p_readout=0.02) are illustrative
NISQ-like values in the range commonly reported for near-term superconducting
hardware, NOT calibrated to any specific real backend (no real quantum
hardware is used anywhere in this study, per this task's own constraint).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data_pipeline as dp
from qnac_core import IDSConfigEnv, run_actor_critic, ABLATION_CONFIGS
from ablation_ladder import _final_eval
import noise_sim as ns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS_DIR / "noise_robustness.csv"

CELLS = [
    ("UNSW-NB15", "A", lambda seed: dp.get_protocol_a(dp.load_unsw_raw, seed)),
    ("CIC-IDS-2017", "A", dp.get_cic_ids2017_protocol_a),
    ("CICIoT23", "A", dp.get_ciciot23_protocol_a),
]
SEEDS = [42, 43]
P_1Q, P_2Q, P_READOUT = 0.01, 0.03, 0.02
N_TRAJ = 200

CONDITIONS = [
    ("ideal", None), ("finite_shot", None),
    ("depolarizing", None), ("readout", None), ("readout", "readout"),
    ("combined", None), ("combined", "readout"), ("combined", "zne"),
]


def kl_div(p, q):
    p = np.clip(p, 1e-12, 1); q = np.clip(q, 1e-12, 1)
    return float(np.sum(p * np.log(p / q)))


def main():
    rows = []
    for dataset, protocol, loader in CELLS:
        for seed in SEEDS:
            t0 = time.time()
            split = loader(seed)
            env = IDSConfigEnv(split["X_train"], split["y_train"], split["X_val"], split["y_val"],
                                split["feature_names"], n_groups=8, seed=seed)
            res = run_actor_critic(env, budget=60, seed=seed, n_qubits=4, layers=2,
                                    **ABLATION_CONFIGS["7_sf_qnac_ids"])
            actor, s_final = res["actor"], res["final_state"]
            circ = actor.build_circuit_for_state(s_final)
            p_ideal, z_ideal = actor.policy(s_final)
            a_ideal = int(np.argmax(p_ideal))
            print(f"[{dataset}/{protocol}/seed={seed}] trained in {time.time()-t0:.1f}s, "
                  f"ideal action={a_ideal}, best_val_f1={res['best_val_f1_macro']:.4f}")

            for condition, mitigation in CONDITIONS:
                rng = np.random.default_rng(seed + 7000)
                if condition == "finite_shot":
                    from quantum_sim import shot_noisy_expectation
                    z = shot_noisy_expectation(z_ideal, 1024, rng)
                else:
                    z = ns.combined_noise_z(circ, condition, rng, p_1q=P_1Q, p_2q=P_2Q,
                                             p_readout=P_READOUT, n_trajectories=N_TRAJ,
                                             mitigation=mitigation)
                p_noisy = actor.policy_from_z(z)
                a_noisy = int(np.argmax(p_noisy))
                action_changed = a_noisy != a_ideal

                # downstream impact: replay env from the trained final state's
                # underlying config with the noise-selected action, full eval
                env2 = IDSConfigEnv(split["X_train"], split["y_train"], split["X_val"], split["y_val"],
                                     split["feature_names"], n_groups=8, seed=seed)
                env2.active = env.active.copy(); env2.hp = dict(env.hp)
                _, metrics_noisy, _, mask_noisy = env2.step(a_noisy, fidelity="full")
                downstream = _final_eval(split["X_train"], split["y_train"], split["X_test"],
                                          split["y_test"], mask_noisy, env2.hp, seed=seed)

                rows.append({
                    "dataset": dataset, "protocol": protocol, "seed": seed,
                    "condition": condition, "mitigation": mitigation or "none",
                    "kl_div_from_ideal": kl_div(p_noisy, p_ideal),
                    "action_changed": action_changed,
                    "ideal_action": a_ideal, "noisy_action": a_noisy,
                    "downstream_test_f1_macro": downstream["test_f1_macro"],
                    "downstream_test_fpr": downstream["test_fpr"],
                })
                print(f"    {condition:14s} mit={str(mitigation):8s} "
                      f"KL={rows[-1]['kl_div_from_ideal']:.4f} action_changed={action_changed} "
                      f"downstream_f1={downstream['test_f1_macro']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
