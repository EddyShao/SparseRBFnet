#!/usr/bin/env python3
import os
import re
import pickle
import json
import math
from typing import Dict, Any, List, Optional
import sys
sys.path.insert(0, '.')

BASE_DIR = "exps/exp_bi/"    # change this if needed
OUTPUT_JSON = os.path.join(BASE_DIR, "aggregated_results.json")




def extract_seed_from_filename(filename: str) -> Optional[int]:
    """
    Try to parse seed from something like 'bilap_gaussian_low_seed_200.pkl'.
    Returns None if no seed could be parsed.
    """
    m = re.search(r"seed[_\-]?(\d+)", filename)
    if m:
        return int(m.group(1))
    return None


def compute_total_runtime(experiment_obj) -> float:
    """
    Sum runtime over all levels and phases.
    This is more reasonable than just taking one phase.
    """
    total = 0.0
    if experiment_obj is None:
        return 0.0

    # experiment_obj is an ExperimentResult: has .levels (list[LevelResult])
    for level in getattr(experiment_obj, "levels", []):
        for phase in getattr(level, "phases", []):
            meta = getattr(phase.output, "meta", {})
            rt = meta.get("runtime", 0.0)
            total += float(rt)
    return float(total)


def aggregate_one_pkl(path: str) -> Dict[str, Any]:
    """
    Load one .pkl file and extract:
      - rel_L_2_test
      - final_supp
      - runtime
      - seed
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # --- summary ---
    summary_raw = data["summary"]
    s_last = summary_raw[1]

    # Adapt keys if needed (you can add fallbacks here)
    rel_L2 = s_last.get("rel_L_2_test", None)
    final_supp = s_last.get("final_supp", None)

    # --- runtime from ExperimentResult ---
    experiment_obj = data.get("experiment", None)
    runtime = compute_total_runtime(experiment_obj)

    # --- seed from filename ---
    filename = os.path.basename(path)
    seed = extract_seed_from_filename(filename)

    return {
        "path": path,
        "seed": seed,
        "rel_L_2_test": rel_L2,
        "final_supp": final_supp,
        "runtime": runtime,
    }


def mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan}
    n = len(values)
    mu = sum(values) / n
    if n == 1:
        return {"mean": mu, "std": 0.0}
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return {"mean": mu, "std": math.sqrt(var)}


def main():
    # experiment_name -> list of run dicts
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    # Walk through BASE_DIR and collect all *_results folders
    for root, dirs, files in os.walk(BASE_DIR):
        for d in dirs:
            if not d.endswith("_results"):
                continue
            exp_name = d[:-len("_results")]  # strip suffix
            full_dir = os.path.join(root, d)

            # collect pkl files
            pkl_files = [
                os.path.join(full_dir, f)
                for f in os.listdir(full_dir)
                if f.endswith(".pkl")
            ]

            if not pkl_files:
                continue

            print(f"Processing experiment '{exp_name}' in {full_dir} "
                  f"({len(pkl_files)} .pkl files)")

            runs = []
            for pkl_path in sorted(pkl_files):
                try:
                    run_info = aggregate_one_pkl(pkl_path)
                    runs.append(run_info)
                except Exception as e:
                    print(f"  [WARN] Failed to process {pkl_path}: {e}")

            if runs:
                all_results.setdefault(exp_name, []).extend(runs)

    # Build aggregated summary
    aggregated: Dict[str, Any] = {}

    for exp_name, runs in all_results.items():
        rels = [r["rel_L_2_test"] for r in runs if r["rel_L_2_test"] is not None]
        sups = [r["final_supp"] for r in runs if r["final_supp"] is not None]
        rts = [r["runtime"] for r in runs if r["runtime"] is not None]

        aggregated[exp_name] = {
            "num_runs": len(runs),
            "rel_L_2_test": mean_std(rels),
            "final_supp": mean_std(sups),
            "runtime": mean_std(rts),
            "runs": runs,  # keep per-run info for reference
        }

    # Save to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(aggregated, f, indent=2, sort_keys=True)

    print(f"\nSaved aggregated results to {OUTPUT_JSON}")
    print("Experiments summarized:", ", ".join(sorted(aggregated.keys())))


if __name__ == "__main__":
    main()