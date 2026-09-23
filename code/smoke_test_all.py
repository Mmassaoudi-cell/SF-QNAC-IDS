import time
import numpy as np
import data_pipeline as dp
import ablation_ladder as al
import benchmarks as bm

print("Loading a small UNSW-NB15 Protocol A split...")
split = dp.get_protocol_a(dp.load_unsw_raw, seed=42, working_n=1500)
print({k: (v.shape if hasattr(v, "shape") else v) for k, v in split.items() if k != "feature_names"})

t0 = time.time()
print("\n--- rung 1 xgboost ---")
print(al.run_rung1_xgboost(split, seed=42))

print("\n--- rung 2 sho_xgboost (tiny budget) ---")
print(al.run_rung2_sho_xgboost(split, seed=42, n_horses=3, n_iter=2, search_n=500))

print("\n--- rung 3 qrl_sho_xgboost_repro (tiny budget) ---")
import baseline_repro as br
br.SHO_N, br.SHO_T, br.RL_BUDGET = 3, 2, 6
print(al.run_rung3_qrl_sho_xgboost_repro(split, seed=42))

print("\n--- rungs 4-7 actor-critic (tiny budget) ---")
for name in ["4_classical_actor_critic", "5_vqc_actor_critic_ordinary_grad",
             "6_qnac_fixed_shots", "7_sf_qnac_ids"]:
    r = al.run_actor_critic_rung(name, split, seed=42, budget=8, n_qubits=4, layers=2)
    print(name, "->", {k: r[k] for k in ["test_f1_macro", "n_qubits", "total_full_trainings",
                                          "total_shots", "total_circuit_evals",
                                          "episodes_to_convergence", "wall_time_total_sec"]})

print("\n--- benchmarks ---")
for name, fn in bm.BENCHMARK_RUNNERS.items():
    if name == "1_xgboost_default":
        continue
    kwargs = dict(n_trials=5) if name == "bench_optuna_xgboost" else {}
    r = fn(split["X_train"], split["y_train"], split["X_val"], split["y_val"],
           split["X_test"], split["y_test"], seed=42, **kwargs)
    print(name, "->", {k: r[k] for k in ["test_f1_macro", "wall_time_total_sec", "total_full_trainings"]})

print("\nTOTAL SMOKE TEST TIME", time.time() - t0)
