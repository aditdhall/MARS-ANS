import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from evaluate import run_ablation

ROOT      = os.path.dirname(os.path.dirname(__file__))
DQN_PATH  = os.path.join(ROOT, 'models', 'dqn_rover.pt')
OUT_DIR   = os.path.join(ROOT, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

results = {"alpha_sweep": {}, "h_crit_sweep": {}}

print("=== Ablation 1: alpha sweep ===")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    r = run_ablation(n_runs=20, alpha=alpha, h_crit=0.7, dqn_path=DQN_PATH)
    print(f"alpha={alpha}: {r}")
    results["alpha_sweep"][str(alpha)] = r

print("\n=== Ablation 2: h_crit sweep ===")
for h_crit in [0.4, 0.5, 0.6, 0.7, 0.8]:
    r = run_ablation(n_runs=20, alpha=0.75, h_crit=h_crit, dqn_path=DQN_PATH)
    print(f"h_crit={h_crit}: {r}")
    results["h_crit_sweep"][str(h_crit)] = r

out_path = os.path.join(ROOT, "ablation_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {out_path}")

print(f"\n{'config':<20} {'completion':>12} {'avg_steps':>12} {'avg_reward':>12}")
print("-" * 60)
for alpha, r in results["alpha_sweep"].items():
    print(f"{'alpha='+alpha:<20} {r['completion_rate']:>12.3f} {r['avg_steps']:>12.2f} {r['avg_reward']:>12.2f}")
for h_crit, r in results["h_crit_sweep"].items():
    print(f"{'h_crit='+h_crit:<20} {r['completion_rate']:>12.3f} {r['avg_steps']:>12.2f} {r['avg_reward']:>12.2f}")
