import json
import numpy as np
import copy


def reprocess_experiment(exp,
                         key_l2="rel_L_2_test",
                         key_linf="L_inf_test",
                         factor=3.0):
    runs = exp["runs"]

    # extract errors
    l2 = np.array([r[key_l2] for r in runs], dtype=float)
    linf = np.array([r[key_linf] for r in runs], dtype=float)

    l2_min = float(l2.min())
    linf_min = float(linf.min())

    l2_th = factor * l2_min
    linf_th = factor * linf_min

    # keep run only if it satisfies BOTH conditions
    kept = [
        r for r in runs
        if (r[key_l2] <= l2_th) and (r[key_linf] <= linf_th)
    ]

    def stats(vals):
        vals = np.asarray(vals, dtype=float)
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0))
        }

    new_exp = copy.deepcopy(exp)
    new_exp["num_runs_raw"] = len(runs)
    new_exp["num_runs"] = len(kept)

    new_exp["filter_rule"] = (
        f"({key_linf} <= {factor} * min({key_linf})) AND "
        f"({key_l2} <= {factor} * min({key_l2}))"
    )
    new_exp["thresholds"] = {
        key_linf: float(linf_th),
        key_l2: float(l2_th),
    }

    new_exp["runs"] = kept

    # recompute summary stats from kept runs
    new_exp[key_l2] = stats([r[key_l2] for r in kept]) if kept else {"mean": float("nan"), "std": float("nan")}
    new_exp[key_linf] = stats([r[key_linf] for r in kept]) if kept else {"mean": float("nan"), "std": float("nan")}
    new_exp["final_supp"] = stats([r["final_supp"] for r in kept]) if kept else {"mean": float("nan"), "std": float("nan")}
    new_exp["runtime"] = stats([r["runtime"] for r in kept]) if kept else {"mean": float("nan"), "std": float("nan")}

    return new_exp


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    in_path = "exps/exp_Eikonal_1D/aggregated_results.json"
    out_path = "exps/exp_Eikonal_1D/aggregated_results_cleaned.json"

    with open(in_path, "r") as f:
        data = json.load(f)

    cleaned = {name: reprocess_experiment(exp) for name, exp in data.items()}

    with open(out_path, "w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"Saved cleaned results to {out_path}")