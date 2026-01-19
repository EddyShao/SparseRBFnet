# import os
# import re
# import json
# import pickle
# import argparse
# from collections import defaultdict

# import numpy as np
# import matplotlib.pyplot as plt

# import matplotlib as mpl

# mpl.rcParams.update({
#     # --- font ---
#     "font.family": "serif",
#     "font.serif": ["STIXGeneral"],      # Times-like, always available
#     "mathtext.fontset": "stix",         # math matches text
#     "axes.unicode_minus": False,

#     # --- sizes ---
#     "font.size": 14,
#     "axes.labelsize": 16,
#     "axes.titlesize": 16,
#     "legend.fontsize": 14,
#     "xtick.labelsize": 13,
#     "ytick.labelsize": 13,

#     # --- lines ---
#     "lines.linewidth": 2.5,
# })


# KEY_RE = re.compile(r"eikonal_eps(\d+)_N_(\d+)")


# def parse_eps(eps_digits: str) -> float:
#     """
#     Parse eps string like '0020' into epsilon.
#     Convention: epsXXXX means epsilon = XXXX / 1e4, so eps0020 -> 0.0020.

#     If your naming convention differs, change the scale below.
#     """
#     return 1 / int(eps_digits)


# def load_errors_from_json(json_path: str):
#     """
#     Returns:
#       L2_by_N[N][eps] = mean rel_L_2_test
#       Linf_by_N[N][eps] = mean L_inf_test
#       L2_std_by_N[N][eps] = std rel_L_2_test (optional)
#       Linf_std_by_N[N][eps] = std L_inf_test (optional)
#     """
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     L2_by_N = defaultdict(dict)
#     Linf_by_N = defaultdict(dict)
#     L2_std_by_N = defaultdict(dict)
#     Linf_std_by_N = defaultdict(dict)

#     bad_keys = 0

#     for k, v in data.items():
#         m = KEY_RE.match(k)
#         if not m:
#             bad_keys += 1
#             continue

#         eps = parse_eps(m.group(1))
#         N = int(m.group(2))

#         # Prefer summary stats if present
#         if "rel_L_2_test" in v and "mean" in v["rel_L_2_test"]:
#             L2_by_N[N][eps] = float(v["rel_L_2_test"]["mean"])
#             L2_std_by_N[N][eps] = float(v["rel_L_2_test"].get("std", 0.0))
#         else:
#             # Fallback: compute from runs
#             runs = v.get("runs", [])
#             vals = [r["rel_L_2_test"] for r in runs if "rel_L_2_test" in r]
#             if vals:
#                 L2_by_N[N][eps] = float(np.mean(vals))
#                 L2_std_by_N[N][eps] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

#         if "L_inf_test" in v and "mean" in v["L_inf_test"]:
#             Linf_by_N[N][eps] = float(v["L_inf_test"]["mean"])
#             Linf_std_by_N[N][eps] = float(v["L_inf_test"].get("std", 0.0))
#         else:
#             runs = v.get("runs", [])
#             vals = [r["L_inf_test"] for r in runs if "L_inf_test" in r]
#             if vals:
#                 Linf_by_N[N][eps] = float(np.mean(vals))
#                 Linf_std_by_N[N][eps] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

#     if bad_keys > 0:
#         print(f"[warn] Skipped {bad_keys} keys that did not match pattern {KEY_RE.pattern}")

#     # Drop any N that ended up empty
#     L2_by_N = {N: d for N, d in L2_by_N.items() if len(d) > 0}
#     Linf_by_N = {N: d for N, d in Linf_by_N.items() if len(d) > 0}
#     L2_std_by_N = {N: d for N, d in L2_std_by_N.items() if len(d) > 0}
#     Linf_std_by_N = {N: d for N, d in Linf_std_by_N.items() if len(d) > 0}

#     return L2_by_N, Linf_by_N, L2_std_by_N, Linf_std_by_N


# def maybe_remove_eps(d_by_N, eps_to_remove):
#     if not eps_to_remove:
#         return d_by_N
#     out = {}
#     for N, d in d_by_N.items():
#         dd = dict(d)
#         for e in eps_to_remove:
#             if e in dd:
#                 del dd[e]
#         out[N] = dd
#     return out


# def plot_error_vs_eps(
#     errors_by_N,
#     std_by_N,
#     out_path,
#     ylabel,
#     title,
#     use_errorbar=False,
#     domain_length=2.0,
#     keep_N=6,              # e.g. 4 -> only plot 4 smallest N; None -> all
#     keep_eps=None,            # e.g. 6 -> only keep 6 eps per N; None -> all
#     eps_keep="largest",      # "smallest" or "largest"
# ):
#     """
#     Plot error vs epsilon for each N.

#     NEW (simple) controls:
#       - keep_N: keep only first keep_N values of N (sorted ascending)
#       - keep_eps: keep only keep_eps epsilon points per curve
#       - eps_keep: choose which eps points to keep ("smallest" or "largest")
#     """
#     plt.figure(figsize=(10, 6))

#     Ns = sorted(errors_by_N.keys())
#     if keep_N is not None:
#         Ns = Ns[:keep_N]
    
#     # <<< ADD THIS >>>
#     print_stats_table(
#         {N: errors_by_N[N] for N in Ns},
#         {N: std_by_N.get(N, {}) for N in Ns},
#         ylabel,
#     )

#     for N in Ns:
#         eps_vals = sorted(errors_by_N[N].keys())
#         if not eps_vals:
#             continue

#         # Keep only some eps values if requested
#         if keep_eps is not None:
#             if eps_keep == "smallest":
#                 eps_vals = eps_vals[:keep_eps]
#             elif eps_keep == "largest":
#                 eps_vals = eps_vals[-keep_eps:]
#             else:
#                 raise ValueError(f"eps_keep must be 'smallest' or 'largest', got {eps_keep}")

#         means = [errors_by_N[N][e] for e in eps_vals]
#         h = domain_length / N

#         if use_errorbar:
#             yerr = [std_by_N.get(N, {}).get(e, 0.0) for e in eps_vals]
#             plt.errorbar(
#                 eps_vals, means, yerr=yerr,
#                 marker="o", capsize=3, linestyle="-",
#                 label=fr"$N = {N}$, $h=\frac{{2}}{{{N}}}$"
#                 )
#             plt.xscale("log")
#             plt.yscale("log")
#         else:
#             plt.loglog(eps_vals, means, marker="o", label=fr"$N = {N}$, $h=\frac{{2}}{{{N}}}$")

#     plt.xlabel(r"$\epsilon$")
#     plt.ylabel(ylabel)
#     # plt.title(title)
#     plt.grid(True, which="both", ls="--", alpha=0.5)
#     plt.legend()
#     plt.tight_layout()

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     print(f"Plot saved to {out_path}")


# def print_stats_table(errors_by_N, std_by_N, label):
#     """
#     Print mean ± std for each (N, eps) pair.
#     Intended for quick inspection and LaTeX table preparation.
#     """
#     print("\n" + "=" * 80)
#     print(f"Statistics for {label}")
#     print("=" * 80)

#     for N in sorted(errors_by_N.keys()):
#         for eps in sorted(errors_by_N[N].keys()):
#             mean = errors_by_N[N][eps]
#             std = std_by_N.get(N, {}).get(eps, 0.0)
#             print(
#                 f"N={N:4d}, eps={eps:10.4e} : "
#                 f"{mean:.4e} ± {std:.2e}"
#             )


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--json", default='exps/exp_Eikonal_1D/aggregated_results_cleaned.json',help="Path to the JSON summary file.")
#     parser.add_argument("--base_dir", default='exps/exp_Eikonal_1D/',
#                         help="Output directory (default: directory of the JSON file).")
#     parser.add_argument("--no_errorbar", action="store_true",
#                         help="Disable mean±std error bars (default: on).")
#     parser.add_argument("--drop_eps", nargs="*", type=float, default=[],
#                         help="Eps values to remove, e.g. --drop_eps 0.1 0.05")
#     parser.add_argument("--save_pkl", action="store_true",
#                         help="Also save extracted dicts as PKL for reuse.")
#     args = parser.parse_args()

#     json_path = args.json
#     base_dir = args.base_dir if args.base_dir is not None else os.path.dirname(json_path)

#     L2_by_N, Linf_by_N, L2_std_by_N, Linf_std_by_N = load_errors_from_json(json_path)

#     # Optional removal of large eps for cleaner viz (like your old script)
#     if args.drop_eps:
#         L2_by_N = maybe_remove_eps(L2_by_N, args.drop_eps)
#         Linf_by_N = maybe_remove_eps(Linf_by_N, args.drop_eps)
#         L2_std_by_N = maybe_remove_eps(L2_std_by_N, args.drop_eps)
#         Linf_std_by_N = maybe_remove_eps(Linf_std_by_N, args.drop_eps)

#     use_errorbar = not args.no_errorbar

#     # Plot L2
#     plot_error_vs_eps(
#         errors_by_N=L2_by_N,
#         std_by_N=L2_std_by_N,
#         out_path=os.path.join(base_dir, "eikonal_l2_vs_eps_h.png"),
#         ylabel=r"Relative $L_2$ error",
#         title=r"Eikonal (1D): relative $L_2$ error vs $\varepsilon$ for different $N$",
#         use_errorbar=use_errorbar,
#         domain_length=2.0,
#     )

#     # Plot Linf
#     plot_error_vs_eps(
#         errors_by_N=Linf_by_N,
#         std_by_N=Linf_std_by_N,
#         out_path=os.path.join(base_dir, "eikonal_linf_vs_eps_h.png"),
#         ylabel=r"$L^\infty$ error (averaged)",
#         title=r"Eikonal (1D): $L^\infty$ error vs $\epsilon$ for different $N$",
#         use_errorbar=use_errorbar,
#         domain_length=2.0,
#     )



# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# Paper-style rcParams
# =========================
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

KEY_RE = re.compile(r"eikonal_eps(\d+)_N_(\d+)")

def parse_eps(eps_digits: str) -> float:
    """
    Parse eps string like '0020' into epsilon.
    Current convention: epsXXXX means epsilon = 1 / XXXX (as in your code).
    Change here if needed.
    """
    return 1.0 / int(eps_digits)

def load_errors_from_json(json_path: str):
    """
    Returns:
      L2_by_N[N][eps] = mean rel_L_2_test
      Linf_by_N[N][eps] = mean L_inf_test
      L2_std_by_N[N][eps] = std rel_L_2_test
      Linf_std_by_N[N][eps] = std L_inf_test
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    L2_by_N = defaultdict(dict)
    Linf_by_N = defaultdict(dict)
    L2_std_by_N = defaultdict(dict)
    Linf_std_by_N = defaultdict(dict)

    bad_keys = 0

    for k, v in data.items():
        m = KEY_RE.match(k)
        if not m:
            bad_keys += 1
            continue

        eps = parse_eps(m.group(1))
        N = int(m.group(2))

        # L2
        if "rel_L_2_test" in v and isinstance(v["rel_L_2_test"], dict) and "mean" in v["rel_L_2_test"]:
            L2_by_N[N][eps] = float(v["rel_L_2_test"]["mean"])
            L2_std_by_N[N][eps] = float(v["rel_L_2_test"].get("std", 0.0))
        else:
            runs = v.get("runs", [])
            vals = [r.get("rel_L_2_test") for r in runs if r.get("rel_L_2_test") is not None]
            if vals:
                L2_by_N[N][eps] = float(np.mean(vals))
                L2_std_by_N[N][eps] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        # Linf
        if "L_inf_test" in v and isinstance(v["L_inf_test"], dict) and "mean" in v["L_inf_test"]:
            Linf_by_N[N][eps] = float(v["L_inf_test"]["mean"])
            Linf_std_by_N[N][eps] = float(v["L_inf_test"].get("std", 0.0))
        else:
            runs = v.get("runs", [])
            vals = [r.get("L_inf_test") for r in runs if r.get("L_inf_test") is not None]
            if vals:
                Linf_by_N[N][eps] = float(np.mean(vals))
                Linf_std_by_N[N][eps] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    if bad_keys > 0:
        print(f"[warn] Skipped {bad_keys} keys that did not match pattern {KEY_RE.pattern}")

    # Drop empty Ns
    L2_by_N = {N: d for N, d in L2_by_N.items() if d}
    Linf_by_N = {N: d for N, d in Linf_by_N.items() if d}
    L2_std_by_N = {N: d for N, d in L2_std_by_N.items() if d}
    Linf_std_by_N = {N: d for N, d in Linf_std_by_N.items() if d}

    return L2_by_N, Linf_by_N, L2_std_by_N, Linf_std_by_N

def maybe_remove_eps(d_by_N, eps_to_remove):
    if not eps_to_remove:
        return d_by_N
    out = {}
    for N, d in d_by_N.items():
        dd = dict(d)
        for e in eps_to_remove:
            if e in dd:
                del dd[e]
        out[N] = dd
    return out

def print_stats_table(errors_by_N, std_by_N, label):
    print("\n" + "=" * 80)
    print(f"Statistics for {label}")
    print("=" * 80)
    for N in sorted(errors_by_N.keys()):
        for eps in sorted(errors_by_N[N].keys()):
            mean = errors_by_N[N][eps]
            std = std_by_N.get(N, {}).get(eps, 0.0)
            print(f"N={N:4d}, eps={eps:10.4e} : {mean:.4e} ± {std:.2e}")

def plot_error_vs_eps(
    errors_by_N,
    std_by_N,
    out_path,
    ylabel,
    title,
    use_errorbar=False,
    domain_length=2.0,
    keep_N=6,
    keep_eps=None,
    eps_keep="largest",
):
    """
    Plot error vs epsilon for each N.
    Each curve gets a different marker (circle/triangle/square/...)
    """
    MARKERS = ["o", "s", "^", "D", "v", ">", "<", "P", "X", "*", "h", "p"]
    plt.figure(figsize=(10, 6))

    Ns = sorted(errors_by_N.keys())
    if keep_N is not None:
        Ns = Ns[:keep_N]

    # print table for the plotted subset
    print_stats_table(
        {N: errors_by_N[N] for N in Ns},
        {N: std_by_N.get(N, {}) for N in Ns},
        ylabel,
    )

    for i, N in enumerate(Ns):
        marker = MARKERS[i % len(MARKERS)]
        eps_vals = sorted(errors_by_N[N].keys())
        if not eps_vals:
            continue

        if keep_eps is not None:
            if eps_keep == "smallest":
                eps_vals = eps_vals[:keep_eps]
            elif eps_keep == "largest":
                eps_vals = eps_vals[-keep_eps:]
            else:
                raise ValueError(f"eps_keep must be 'smallest' or 'largest', got {eps_keep}")

        means = [errors_by_N[N][e] for e in eps_vals]

        if use_errorbar:
            yerr = [std_by_N.get(N, {}).get(e, 0.0) for e in eps_vals]
            plt.errorbar(
                eps_vals, means, yerr=yerr,
                marker=marker, capsize=3, linestyle="-",
                label=fr"$N = {N}$, $h=\frac{{{domain_length}}}{{{N}}}$"
            )
            plt.xscale("log")
            plt.yscale("log")
        else:
            plt.loglog(
                eps_vals, means,
                marker=marker, linestyle="-",
                label=fr"$N = {N}$, $h=\frac{{{domain_length}}}{{{N}}}$"
            )

    plt.xlabel(r"$\epsilon$")
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Plot saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="exps/exp_Eikonal_1D/aggregated_results_cleaned.json",
                        help="Path to the JSON summary file.")
    parser.add_argument("--base_dir", default="exps/exp_Eikonal_1D/",
                        help="Output directory (default: directory of the JSON file).")
    parser.add_argument("--no_errorbar", action="store_true",
                        help="Disable mean±std error bars (default: on).")
    parser.add_argument("--drop_eps", nargs="*", type=float, default=[],
                        help="Eps values to remove, e.g. --drop_eps 0.1 0.05")
    parser.add_argument("--keep_N", type=int, default=6,
                        help="Keep only the smallest keep_N values of N (sorted). Use -1 for all.")
    parser.add_argument("--keep_eps", type=int, default=None,
                        help="Keep only keep_eps epsilon points per curve. Default: all.")
    parser.add_argument("--eps_keep", choices=["smallest", "largest"], default="largest",
                        help="If keep_eps is set, choose which eps points to keep.")
    args = parser.parse_args()

    json_path = args.json
    base_dir = args.base_dir if args.base_dir is not None else os.path.dirname(json_path)

    L2_by_N, Linf_by_N, L2_std_by_N, Linf_std_by_N = load_errors_from_json(json_path)

    if args.drop_eps:
        L2_by_N = maybe_remove_eps(L2_by_N, args.drop_eps)
        Linf_by_N = maybe_remove_eps(Linf_by_N, args.drop_eps)
        L2_std_by_N = maybe_remove_eps(L2_std_by_N, args.drop_eps)
        Linf_std_by_N = maybe_remove_eps(Linf_std_by_N, args.drop_eps)

    use_errorbar = not args.no_errorbar
    keep_N = None if args.keep_N == -1 else args.keep_N

    plot_error_vs_eps(
        errors_by_N=L2_by_N,
        std_by_N=L2_std_by_N,
        out_path=os.path.join(base_dir, "eikonal_l2_vs_eps_h.png"),
        ylabel=r"Relative $L_2$ error",
        title=r"Eikonal (1D): relative $L_2$ error vs $\varepsilon$ for different $N$",
        use_errorbar=use_errorbar,
        domain_length=2.0,
        keep_N=keep_N,
        keep_eps=args.keep_eps,
        eps_keep=args.eps_keep,
    )

    plot_error_vs_eps(
        errors_by_N=Linf_by_N,
        std_by_N=Linf_std_by_N,
        out_path=os.path.join(base_dir, "eikonal_linf_vs_eps_h.png"),
        ylabel=r"$L^\infty$ error (averaged)",
        title=r"Eikonal (1D): $L^\infty$ error vs $\epsilon$ for different $N$",
        use_errorbar=use_errorbar,
        domain_length=2.0,
        keep_N=keep_N,
        keep_eps=args.keep_eps,
        eps_keep=args.eps_keep,
    )

if __name__ == "__main__":
    main()