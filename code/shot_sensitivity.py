"""
Phase 2: shot-sensitivity sweep. Trains the QNAC configuration (natural
gradient, but FIXED shots throughout -- i.e. Sec.10's "8192 shots" baseline
behavior generalized to any fixed shot count, deliberately NOT the adaptive
SF-QNAC-IDS variant, so the sweep isolates the effect of shot count alone)
at shot levels {64, 128, 256, 512, 1024, 2048, 4096, 8192}, and measures:
  - final test Macro-F1 (does accuracy hold up at low shot counts?)
  - policy stability: KL-divergence and argmax-agreement between the
    trained actor's ideal (0-shot/exact) policy and its N-shot sampled
    policy, both evaluated at the actor's own final state
  - total training wall-clock time
Bounded to 2 datasets x 2 seeds (8 shot levels x 2 x 2 = 32 runs) for
compute reasons -- a stated Phase-2-within-Phase-2 scope reduction from
"all datasets, many seeds," documented here rather than silently expanded.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data_pipeline as dp
from qnac_core import IDSConfigEnv, run_actor_critic, ABLATION_CONFIGS
from ablation_ladder import _final_eval
from quantum_sim import shot_noisy_expectation

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS_DIR / "shot_sensitivity.csv"

SHOT_LEVELS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
CELLS = [
    ("UNSW-NB15", "A", lambda seed: dp.get_protocol_a(dp.load_unsw_raw, seed)),
    ("CICIoT23", "A", dp.get_ciciot23_protocol_a),
]
SEEDS = [42, 43]


def kl_div(p, q):
    p = np.clip(p, 1e-12, 1); q = np.clip(q, 1e-12, 1)
    return float(np.sum(p * np.log(p / q)))


def main():
    rows = []
    for dataset, protocol, loader in CELLS:
        for seed in SEEDS:
            split = loader(seed)
            for shots in SHOT_LEVELS:
                t0 = time.time()
                env = IDSConfigEnv(split["X_train"], split["y_train"], split["X_val"], split["y_val"],
                                    split["feature_names"], n_groups=8, seed=seed)
                cfg = dict(ABLATION_CONFIGS["6_qnac_fixed_shots"])  # natural grad, fixed shots
                res = run_actor_critic(env, budget=60, seed=seed, n_qubits=4, layers=2,
                                        shots_fixed_override=shots, **cfg)
                wall_time = time.time() - t0

                actor, s_final = res["actor"], res["final_state"]
                p_ideal, z_ideal = actor.policy(s_final)
                rng = np.random.default_rng(seed + 9000)
                z_shot = shot_noisy_expectation(z_ideal, shots, rng)
                p_shot = actor.policy_from_z(z_shot)
                kl = kl_div(p_shot, p_ideal)
                agree = int(np.argmax(p_shot)) == int(np.argmax(p_ideal))

                hp = res["best_hp"] or dict(n_estimators=150, max_depth=5, learning_rate=0.1,
                                             min_child_weight=1, gamma=0.0, subsample=0.9,
                                             colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0)
                mask = res["best_mask"] if res["best_mask"] is not None else np.ones(split["X_train"].shape[1], dtype=bool)
                final = _final_eval(split["X_train"], split["y_train"], split["X_test"],
                                     split["y_test"], mask, hp, seed=seed)

                rows.append({
                    "dataset": dataset, "protocol": protocol, "seed": seed, "shots": shots,
                    "test_f1_macro": final["test_f1_macro"], "test_fpr": final["test_fpr"],
                    "kl_div_from_ideal_policy": kl, "argmax_agrees_with_ideal": agree,
                    "episodes_to_convergence": res["episodes_to_convergence"],
                    "wall_time_sec": wall_time,
                })
                print(f"[{dataset}/{protocol}/seed={seed}] shots={shots:5d} "
                      f"F1={final['test_f1_macro']:.4f} KL={kl:.4f} agree={agree} time={wall_time:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
