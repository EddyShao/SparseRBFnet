import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(".")  # to run from repo root

# load previously saved errors if available
# ---- configuration ----
BASE_DIR = "exps/exp_Eikonal"  # change if you are not
ERROR_PATH = os.path.join(BASE_DIR, "eikonal_errors_by_N.pkl")
# load errors if file exists
if os.path.isfile(ERROR_PATH):
    with open(ERROR_PATH, "rb") as f:
        errors_by_N = pickle.load(f)
else:
    print(f"No existing error file found at {ERROR_PATH}. Please run experiments first.")
    raise FileNotFoundError(f"No existing error file found at {ERROR_PATH}.")


# ignore 0.1 and 0.05 for better visualization
for N in errors_by_N:
    if 0.1 in errors_by_N[N]:
        del errors_by_N[N][0.1]
    if 0.05 in errors_by_N[N]:
        del errors_by_N[N][0.05]
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
plt.savefig(os.path.join(BASE_DIR, "eikonal_linf_vs_eps_focus.png"), dpi=300)
print(f"Plot saved to {os.path.join(BASE_DIR, 'eikonal_linf_vs_eps_focus.png')}")