#!/usr/bin/env python3
import os
import re
import pickle
import json
import math
from typing import Dict, Any, List, Optional
import sys

sys.path.insert(0, ".")

BASE_DIR = "exps/exp_bi_pos/"  # change this if needed
OUTPUT_JSON = os.path.join(BASE_DIR, "aggregated_results.json")


def extract_seed_from_filename(filename: str) -> Optional[int]:
    """Parse seed from something like 'bilap_gaussian_low_seed_200.pkl'."""
    m = re.search(r"seed[_\-]?(\d+)", filename)
    return int(m.group(1)) if m else None


def mean_std(values: List[float]) -> Dict[str, float]:
    """Sample mean/std (ddof=1). Filters None/NaN."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": math.nan, "std": math.nan}
    n = len(vals)
    mu = sum(vals) / n
    if n == 1:
        return {"mean": mu, "std": 0.0}
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return {"mean": mu, "std": math.sqrt(var)}


def compute_level_runtimes(experiment_obj) -> List[float]:
    """
    Per-level runtime: runtimes[i] = sum of phase runtimes in level i.
    Expects experiment_obj.levels[*].phases[*].output.meta["runtime"].
    """
    if experiment_obj is None:
        return []

    level_rts: List[float] = []
    for level in getattr(experiment_obj, "levels", []):
        total = 0.0
        for phase in getattr(level, "phases", []):
            out = getattr(phase, "output", None)
            meta = getattr(out, "meta", {}) if out is not None else {}
            rt = meta.get("runtime", 0.0) or 0.0
            total += float(rt)
        level_rts.append(float(total))
    return level_rts


def aggregate_one_pkl(path: str) -> Dict[str, Any]:
    """
    Extract per-level:
      - rel_L_2_test
      - final_supp
      - runtime (per-level)
    Returns:
      {"path":..., "seed":..., "levels":[{level_index,...}, ...]}
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    filename = os.path.basename(path)
    seed = extract_seed_from_filename(filename)

    experiment_obj = data.get("experiment", None)
    level_runtimes = compute_level_runtimes(experiment_obj)

    summary_raw = data.get("summary", None)
    if not isinstance(summary_raw, dict):
        raise ValueError(f"Expected data['summary'] to be a dict, got {type(summary_raw)} in {path}")

    # Deterministic order of summary levels
    keys = sorted(summary_raw.keys())

    levels_out: List[Dict[str, Any]] = []
    for pos, level_index in enumerate(keys):
        level_data = summary_raw[level_index]
        if not isinstance(level_data, dict):
            level_data = {}

        rel_L_2_test = level_data.get("rel_L_2_test", None)
        final_supp = level_data.get("final_supp", None)

        runtime = level_runtimes[pos] if pos < len(level_runtimes) else None

        levels_out.append(
            {
                "level_index": int(level_index),
                "rel_L_2_test": rel_L_2_test,
                "final_supp": final_supp,
                "runtime": runtime,
            }
        )

    # Helpful warning (won't crash)
    if level_runtimes and len(level_runtimes) != len(levels_out):
        print(
            f"[WARN] runtime levels != summary levels in {path}: "
            f"{len(level_runtimes)} vs {len(levels_out)}"
        )

    return {"path": path, "seed": seed, "levels": levels_out}


def main():
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    # Find all *_results folders and collect pkls
    for root, dirs, files in os.walk(BASE_DIR):
        for d in dirs:
            if not d.endswith("_results"):
                continue
            exp_name = d[: -len("_results")]
            full_dir = os.path.join(root, d)

            pkl_files = [
                os.path.join(full_dir, f)
                for f in os.listdir(full_dir)
                if f.endswith(".pkl")
            ]
            if not pkl_files:
                continue

            print(f"Processing '{exp_name}' in {full_dir} ({len(pkl_files)} pkls)")
            runs: List[Dict[str, Any]] = []
            for pkl_path in sorted(pkl_files):
                try:
                    runs.append(aggregate_one_pkl(pkl_path))
                except Exception as e:
                    print(f"  [WARN] Failed {pkl_path}: {e}")

            if runs:
                all_results.setdefault(exp_name, []).extend(runs)

    # Aggregate per experiment, per level_index
    aggregated: Dict[str, Any] = {}
    for exp_name, runs in all_results.items():
        level_rels: Dict[int, List[float]] = {}
        level_sups: Dict[int, List[float]] = {}
        level_rts: Dict[int, List[float]] = {}

        for run in runs:
            for lv in run.get("levels", []):
                li = lv.get("level_index", None)
                if li is None:
                    continue
                li = int(li)

                v = lv.get("rel_L_2_test", None)
                if v is not None:
                    level_rels.setdefault(li, []).append(float(v))

                v = lv.get("final_supp", None)
                if v is not None:
                    level_sups.setdefault(li, []).append(float(v))

                v = lv.get("runtime", None)
                if v is not None:
                    level_rts.setdefault(li, []).append(float(v))

        levels_summary: List[Dict[str, Any]] = []
        for li in sorted(set(level_rels) | set(level_sups) | set(level_rts)):
            levels_summary.append(
                {
                    "level_index": li,
                    "num_runs_with_level": max(
                        len(level_rels.get(li, [])),
                        len(level_sups.get(li, [])),
                        len(level_rts.get(li, [])),
                    ),
                    "rel_L_2_test": mean_std(level_rels.get(li, [])),
                    "final_supp": mean_std(level_sups.get(li, [])),
                    "runtime": mean_std(level_rts.get(li, [])),
                }
            )

        aggregated[exp_name] = {
            "num_runs": len(runs),
            "levels": levels_summary,
            "runs": runs,  # per-run, per-level raw values
        }

    os.makedirs(BASE_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(aggregated, f, indent=2, sort_keys=True)

    print(f"\nSaved aggregated results to {OUTPUT_JSON}")
    if aggregated:
        print("Experiments summarized:", ", ".join(sorted(aggregated.keys())))
    else:
        print("No experiments found. Check BASE_DIR and *_results folders.")


if __name__ == "__main__":
    main()