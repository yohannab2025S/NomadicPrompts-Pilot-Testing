"""One command to regenerate every number and figure in the Part 4 report:  python part4/run_all.py   (from the repo root)
Requires: pip install numpy pandas scipy scikit-learn matplotlib seaborn   Seed is fixed in config.SEED (42)."""
import subprocess, sys
for s in ["merge_logs.py", "analysis_part4.py", "spearman_partA.py", "coding_and_kappa.py", "make_charts.py"]:
    print(f"\n=== {s}"); subprocess.run([sys.executable, f"part4/{s}"], check=True)
