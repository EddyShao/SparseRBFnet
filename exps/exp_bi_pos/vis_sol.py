#!/usr/bin/env python3
import os
import re
import pickle
import json
import math
from typing import Dict, Any, List, Optional
import sys
sys.path.insert(0, '.')
import matplotlib.pyplot as plt

BASE_DIR = "exps/exp_bi_pos/"    # change this if needed
OUTPUT_JSON = os.path.join(BASE_DIR, "aggregated_results.json")

import matplotlib as mpl

mpl.rcParams.update({
    # --- font ---
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],      # Times-like, always available
    "mathtext.fontset": "stix",         # math matches text
    "axes.unicode_minus": False,

    # --- sizes ---
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

    # --- lines ---
    "lines.linewidth": 2.5,
})




def extract_seed_from_filename(filename: str) -> Optional[int]:
    """
    Try to parse seed from something like 'bilap_gaussian_low_seed_200.pkl'.
    Returns None if no seed could be parsed.
    """
    m = re.search(r"seed[_\-]?(\d+)", filename)
    if m:
        return int(m.group(1))
    return None


def level_history(experiment_obj) -> List[float]:
    """
    Return per-level runtime: runtimes[i] = sum of phase runtimes in level i.
    """
    if experiment_obj is None:
        return []

    level_rts: List[float] = []
    for level in getattr(experiment_obj, "levels", []):
        total = 0.0
        for phase in getattr(level, "phases", []):
            out = getattr(phase, "output", None)
            history = getattr(out, "history", {}) if out is not None else {}
        level_rts.append(history)
    return level_rts


def aggregate_one_pkl(path: str) -> Dict[str, Any]:
    """
    Load one .pkl file and extract per-level:
      - rel_L_2_test
      - final_supp
      - runtime
      - seed
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    filename = os.path.basename(path)
    seed = extract_seed_from_filename(filename)

    # Per-level runtime from ExperimentResult
    experiment_obj = data.get("experiment", None)
    history = level_history(experiment_obj)

    first_level = history[0] 
    return first_level['tics'], first_level['js']








file_path_1 = 'exps/exp_bi_pos/bilap_gaussian_outer_results/bilap_gaussian_outer_seed_200.pkl'
file_path_2 = 'exps/exp_bi_pos/bilap_gaussian_results/bilap_gaussian_seed_200.pkl'
inner_tics, inner_js = aggregate_one_pkl(file_path_2)
outer_tics, outer_js = aggregate_one_pkl(file_path_1)
inner_js = [j * 1e-4 for j in inner_js]
outer_js = [j * 1e-4 for j in outer_js]
plt.figure(figsize=(10, 5))
plt.plot(inner_tics, inner_js, label='Full Solver', lw=2.5)
plt.plot(outer_tics, outer_js, label='Outer only', lw=2.5)
plt.xlabel('Time (s)')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig('exps/exp_bi_pos/bi_convergence_tics.png')

plt.clf()

plt.figure(figsize=(10, 5))
plt.plot(list(range(len(inner_tics))), inner_js, label='Full Solver', lw=2.5)
plt.plot(list(range(len(outer_tics))), outer_js, label='Outer only', lw=2.5)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig('exps/exp_bi_pos/bi_convergence_iter.png')