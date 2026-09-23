"""
Phase 2: qubit x depth (layer count) sensitivity grid for the SF-QNAC-IDS
actor. Qubits in {2,4,6,8}, layers in {1,2,3,4} -- 16 configurations. Trains
the full SF-QNAC-IDS config (natural gradient + adaptive shots + cost-aware
reward) at each configuration and records test Macro-F1, trainable quantum
parameter count, circuit depth/gate counts (from the same Circuit.depth()
machinery used throughout, not hand-counted), and wall-clock time -- so a
performance-efficiency Pareto frontier can be read directly off the results,
per the design brief's explicit instruction not to assume more qubits/depth
are better.

Bounded to 2 datasets x 1 seed (16 x 2 = 32 runs) for compute reasons --
stated, not silently expanded to the full spec's larger grid.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data_pipeline as dp
from qnac_core import IDSConfigEnv, run_actor_critic, ABLATION_CONFIGS
from ablation_ladder import _final_eval

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS_DIR / "qubit_depth_sensitivity.csv"

QUBIT_LEVELS = [2, 4, 6, 8]
LAYER_LEVELS = [1, 2, 3, 4]
CELLS = [
    ("UNSW-NB15", "A", lambda seed: dp.get_protocol_a(dp.load_unsw_raw, seed)),
    ("CICIoT23", "A", dp.get_ciciot23_protocol_a),
]
SEED = 42


def main():
    rows = []
    for dataset, protocol, loader in CELLS:
        split = loader(SEED)
        for n_qubits in QUBIT_LEVELS:
            for layers in LAYER_LEVELS:
                t0 = time.time()
                env = IDSConfigEnv(split["X_train"], split["y_train"], split["X_val"], split["y_val"],
                                    split["feature_names"], n_groups=8, seed=SEED)
                res = run_actor_critic(env, budget=60, seed=SEED, n_qubits=n_qubits, layers=layers,
                                        **ABLATION_CONFIGS["7_sf_qnac_ids"])
                wall_time = time.time() - t0
                stats = res["actor_stats"]

                hp = res["best_hp"] or dict(n_estimators=150, max_depth=5, learning_rate=0.1,
                                             min_child_weight=1, gamma=0.0, subsample=0.9,
                                             colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0)
                mask = res["best_mask"] if res["best_mask"] is not None else np.ones(split["X_train"].shape[1], dtype=bool)
                final = _final_eval(split["X_train"], split["y_train"], split["X_test"],
                                     split["y_test"], mask, hp, seed=SEED)

                rows.append({
                    "dataset": dataset, "protocol": protocol, "n_qubits": n_qubits, "layers": layers,
                    "circuit_depth": stats["depth"], "gates_1q": stats["gates_1q"], "gates_2q": stats["gates_2q"],
                    "trainable_params": stats["params"],
                    "test_f1_macro": final["test_f1_macro"], "test_fpr": final["test_fpr"],
                    "total_shots": res["total_shots"], "total_circuit_evals": res["total_circuit_evals"],
                    "episodes_to_convergence": res["episodes_to_convergence"],
                    "wall_time_sec": wall_time,
                })
                print(f"[{dataset}/{protocol}] q={n_qubits} L={layers} depth={stats['depth']} "
                      f"params={stats['params']} F1={final['test_f1_macro']:.4f} time={wall_time:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
