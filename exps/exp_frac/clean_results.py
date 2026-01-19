import json
import numpy as np
import copy


def reprocess_experiment(exp, key="rel_L_2_test", factor=3.0):
    runs = exp["runs"]

    # extract errors
    errors = np.array([r[key] for r in runs])
    emin = errors.min()
    threshold = factor * emin

    # filter runs
    kept = [r for r in runs if r[key] <= threshold]

    def stats(vals):
        vals = np.asarray(vals)
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0))
        }

    new_exp = copy.deepcopy(exp)
    new_exp["num_runs_raw"] = len(runs)
    new_exp["num_runs"] = len(kept)
    new_exp["filter_rule"] = f"{key} <= {factor} * min({key})"
    new_exp["threshold"] = float(threshold)

    new_exp["runs"] = kept
    new_exp["rel_L_2_test"] = stats([r[key] for r in kept])
    new_exp["final_supp"] = stats([r["final_supp"] for r in kept])
    new_exp["runtime"] = stats([r["runtime"] for r in kept])

    return new_exp


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # load original json
    with open("exps/exp_frac/aggregated_results.json", "r") as f:
        data = json.load(f)

    cleaned = {}
    for name, exp in data.items():
        cleaned[name] = reprocess_experiment(exp)

    # save cleaned json
    with open("exps/exp_frac/aggregated_results_cleaned.json", "w") as f:
        json.dump(cleaned, f, indent=2)

    print("Saved cleaned results to exps/exp_frac/aggregated_results_cleaned.json")