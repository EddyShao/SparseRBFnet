import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(".")  # to run from repo root


# ---- configuration ----
BASE_DIR = "exps/exp_Eikonal"  # change if you are not running from that directory
RESULTS_PATTERN = r"eikonal_eps(\d+)_N_(\d+)_results"
ERROR_KEY = "l_inf"  # <-- adjust this if your pickle key is different
# e.g. "err_Linf", "Linf_error", etc.
# ------------------------

# errors_by_N[N][eps] = list of errors over seeds
errors_by_N = {}

for name in os.listdir(BASE_DIR):
    m = re.match(RESULTS_PATTERN, name)
    if not m:
        continue  # skip non-result folders
    else: 
        print(f"Processing results in folder: {name}")
    eps_den_str, N_str = m.groups()
    eps_den = int(eps_den_str)   # e.g. 40 from "0040"
    N = int(N_str)               # e.g. 30 from "030"
    eps = 1.0 / eps_den          # eps = 1 / denominator

    result_dir = os.path.join(BASE_DIR, name)
    if not os.path.isdir(result_dir):
        continue

    # collect errors from all .pkl files (seeds)
    seed_errors = []
    for fname in os.listdir(result_dir):
        if not fname.endswith(".pkl"):
            continue
        fpath = os.path.join(result_dir, fname)
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        try:
            err = data['summary'][1]['L_inf_test']
        except KeyError:
            # helpful debug print – check what keys exist
            print(f"ERROR: key '{ERROR_KEY}' not found in {fpath}")
            print("Available keys:", list(data.keys()))
            raise
        seed_errors.append(float(err))

    if not seed_errors:
        print(f"WARNING: no .pkl files found in {result_dir}")
        continue

    avg_err = float(np.mean(seed_errors))

    if N not in errors_by_N:
        errors_by_N[N] = {}
    errors_by_N[N][eps] = avg_err

# save collected errors for future reference
with open(os.path.join(BASE_DIR, "eikonal_errors_by_N.pkl"), "wb") as f:
    pickle.dump(errors_by_N, f)

# ---- plotting ----
plt.figure()

for N in sorted(errors_by_N.keys()):  # should give 30, 60, 90, 120
    eps_vals = sorted(errors_by_N[N].keys())
    avg_errs = [errors_by_N[N][e] for e in eps_vals]

    # plot error vs eps (eps = 1/den); log-log is typical for convergence
    plt.loglog(eps_vals, avg_errs, marker="o", label=f"N = {N}")
# plt.xscale("log")
plt.xlabel(r"$\varepsilon$")
plt.ylabel(r"$L_\infty$ error (averaged over seeds)")
plt.title(r"Eikonal: $L_\infty$ error vs $\varepsilon$ for different $N$")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "eikonal_linf_vs_eps.png"), dpi=300)
print(f"Plot saved to {os.path.join(BASE_DIR, 'eikonal_linf_vs_eps.png')}")
# plt.show()
# or save:
# plt.savefig("eikonal_linf_vs_eps.png", dpi=300)