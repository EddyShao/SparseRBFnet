#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------
# Matplotlib style
# -----------------------
mpl.rcParams.update({
    # --- font ---
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
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

# -----------------------
# Load JSON
# -----------------------
json_path = "exps/exp_fixed_size/aggregated_results.json"
with open(json_path, "r") as f:
    data = json.load(f)

# -----------------------
# Helpers
# -----------------------
def parse_N(key: str):
    parts = key.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except Exception:
        return None

def is_kind_key(key: str, kind: str):
    """
    kind="fixed"  -> keys like "N_0064" (NOT ending with _smooth or _smooths)
    kind="smooth" -> keys like "N_0064_smooth" OR "N_0064_smooths"
    """
    if kind == "fixed":
        return key.startswith("N_") and (not key.endswith("_smooth")) and (not key.endswith("_smooths"))
    if kind == "smooth":
        return key.startswith("N_") and (key.endswith("_smooth") or key.endswith("_smooths"))
    raise ValueError("kind must be 'fixed' or 'smooth'")

def collect_family(data, *, kind: str, stat_filter=None, success_filter=None):
    """
    Collect rows for a family.

    stat_filter: callable(val)->bool or None
        mean/std/quantiles computed only on vals passing this filter (after finite check).
        If None: compute on all finite values.

    success_filter: callable(val)->bool or None
        success count uses this filter (after finite check).
        If None: success defined as "finite".
    """
    rows = []
    for key, block in data.items():
        if key == "second_order_full":
            continue
        if not is_kind_key(key, kind):
            continue

        N = parse_N(key)
        if N is None:
            continue

        runs = block.get("runs", [])
        n_total = int(len(runs))
        if n_total == 0:
            continue

        raw = np.array([r.get("rel_L_2_test", np.nan) for r in runs], dtype=float)
        finite_mask = np.isfinite(raw)

        # success count
        if success_filter is None:
            success_mask = finite_mask
        else:
            sf = np.vectorize(success_filter, otypes=[bool])
            success_mask = finite_mask & sf(raw)
        n_success = int(np.sum(success_mask))

        # stats values
        vals = raw[finite_mask]
        if stat_filter is not None:
            stf = np.vectorize(stat_filter, otypes=[bool])
            vals = vals[stf(vals)]

        if vals.size == 0:
            mean = std = np.nan
            vmin = q25 = med = q75 = vmax = np.nan
            n_stats = 0
        else:
            mean = float(np.mean(vals))
            std  = float(np.std(vals))
            vmin = float(np.min(vals))
            q25  = float(np.quantile(vals, 0.25))
            med  = float(np.median(vals))
            q75  = float(np.quantile(vals, 0.75))
            vmax = float(np.max(vals))
            n_stats = int(vals.size)

        rows.append({
            "key": key,
            "N": int(N),
            "mean": mean,
            "std": std,
            "n_runs": n_stats,       # number used for stats (post stat_filter)
            "n_success": n_success,  # number passing success_filter
            "n_total": n_total,      # total runs available
            "min": vmin,
            "q25": q25,
            "median": med,
            "q75": q75,
            "max": vmax,
        })

    rows.sort(key=lambda d: d["N"])
    return rows

# -----------------------
# Settings you requested
# -----------------------
SUCCESS_THRESH = 0.30

# fixed: stats computed only on successes
fixed_rows = collect_family(
    data,
    kind="fixed",
    stat_filter=lambda v: v < SUCCESS_THRESH,
    success_filter=lambda v: v < SUCCESS_THRESH,
)

# smooth: ALSO stats computed only on successes (same threshold)
smooth_rows = collect_family(
    data,
    kind="smooth",
    stat_filter=lambda v: v < SUCCESS_THRESH,
    success_filter=lambda v: v < SUCCESS_THRESH,
)

# arrays for plotting
N_fixed      = np.array([d["N"] for d in fixed_rows], dtype=int)
mean_fixed   = np.array([d["mean"] for d in fixed_rows], dtype=float)
std_fixed    = np.array([d["std"]  for d in fixed_rows], dtype=float)

N_smooth     = np.array([d["N"] for d in smooth_rows], dtype=int)
mean_smooth  = np.array([d["mean"] for d in smooth_rows], dtype=float)
std_smooth   = np.array([d["std"]  for d in smooth_rows], dtype=float)

# adaptive benchmark (unchanged)
bench_block = data.get("second_order_full", None)
bench_mean = bench_std = None
if bench_block is not None:
    bench_vals = np.array(
        [r.get("rel_L_2_test", np.nan) for r in bench_block.get("runs", [])],
        dtype=float
    )
    bench_vals = bench_vals[np.isfinite(bench_vals)]
    if bench_vals.size > 0:
        bench_mean = float(np.mean(bench_vals))
        bench_std = float(np.std(bench_vals))

# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(figsize=(10, 6.4))

# adaptive horizontal line + band
if bench_mean is not None:
    ax.axhline(
        bench_mean,
        linestyle="--",
        linewidth=2.2,
        zorder=1,
        label="Our method: adaptive (mean)",
        color="red",
    )
    ax.axhspan(
        bench_mean - bench_std,
        bench_mean + bench_std,
        alpha=0.18,
        zorder=0,
        label=r"Our method: adaptive (mean$\pm$std)",
        color="red",
    )

# fixed: plot only where stats exist (mean/std finite)
mask_fixed = np.isfinite(mean_fixed) & np.isfinite(std_fixed)
ax.plot(
    N_fixed[mask_fixed],
    mean_fixed[mask_fixed],
    marker="s",
    linewidth=2,
    label = (
    fr"Fixed-size network + $\ell^{{1}}$ regularization "
    fr"(mean, "
    fr"$\mathrm{{err}}^{{\mathrm{{rel}}}}_2 \leq {100 * SUCCESS_THRESH:.1f}\%$)"
),
    c="tab:blue",
)
ax.errorbar(
    N_fixed[mask_fixed],
    mean_fixed[mask_fixed],
    yerr=std_fixed[mask_fixed],
    fmt="none",
    elinewidth=1.5,
    capsize=5,
    c="tab:blue",
    label = (
    fr"Fixed-size network + $\ell^{{1}}$ regularization "
    fr"(mean$\pm$std, "
    fr"$\mathrm{{err}}^{{\mathrm{{rel}}}}_2 \leq {100 * SUCCESS_THRESH:.1f}\%$)"
),
)

# annotate fixed success rate
for d in fixed_rows:
    N, y = d["N"], d["mean"]
    if not np.isfinite(y):
        continue
    txt = f"{d['n_success']}/{d['n_total']}"
    ax.annotate(
        txt,
        xy=(N, y),
        xytext=(6, 6),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=12,
        color="tab:blue",
    )

# smooth: plot only where stats exist (mean/std finite)
mask_smooth = np.isfinite(mean_smooth) & np.isfinite(std_smooth)
ax.plot(
    N_smooth[mask_smooth],
    mean_smooth[mask_smooth],
    marker="^",
    linewidth=2,
    linestyle="-.",
    label = (
    fr"Fixed-size network + $\ell^{2}$ regularization "
    fr"(mean, "
    fr"$\mathrm{{err}}^{{\mathrm{{rel}}}}_2 \leq {100 * SUCCESS_THRESH:.1f}\%$)"
),
    c="tab:orange",
)
ax.errorbar(
    N_smooth[mask_smooth],
    mean_smooth[mask_smooth],
    yerr=std_smooth[mask_smooth],
    fmt="none",
    elinewidth=1.5,
    capsize=5,
    c="tab:orange",
    label = (
    fr"Fixed-size network + $\ell^{2}$ regularization "
    fr"(mean$\pm$std, "
    fr"$\mathrm{{err}}^{{\mathrm{{rel}}}}_2 \leq {100 * SUCCESS_THRESH:.1f}\%$)"
),
)

# annotate smooth success rate
for d in smooth_rows:
    N, y = d["N"], d["mean"]
    if not np.isfinite(y):
        continue
    txt = f"{d['n_success']}/{d['n_total']}"
    ax.annotate(
        txt,
        xy=(N, y),
        xytext=(6, -14),  # small offset so it doesn't clash with fixed labels if same N
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=12,
        color="tab:orange",
    )

ax.annotate(
        "10/10",
        xy=(1350, 0.026),
        xytext=(6, -14),  # small offset so it doesn't clash with fixed labels if same N
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=15,
        color="red",
    )


# axes
ax.set_xlim(min(min(N_fixed, default=1), min(N_smooth, default=1)) * 0.8,
            max(max(N_fixed, default=1), max(N_smooth, default=1)) * 1.35)
ax.set_xscale("log", base=2)
ax.set_yscale("log")

xticks = sorted(set(N_fixed.tolist() + N_smooth.tolist()))
ax.set_xticks(xticks)
ax.get_xaxis().set_major_formatter(lambda x, pos: f"{int(x)}")

ax.set_xlabel(r"$N_{\mathrm{fixed}}$")
ax.set_ylabel(r"$\mathrm{err}^{\mathrm{rel}}_{2}$ (relative $L^{2}$ error)")

ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.5)
ax.legend(frameon=True, loc='best', bbox_to_anchor=(0.68, 0.47))

plt.tight_layout()

out_path = "exps/exp_fixed_size/fixed_size_error_vs_N_std.png"
plt.savefig(out_path, dpi=300)
plt.show()

# -----------------------
# Optional print tables
# -----------------------
print("\n[Fixed] (stats computed only for rel_L_2_test < {:.2f})".format(SUCCESS_THRESH))
for d in fixed_rows:
    print(
        f"N={d['N']:4d}  total={d['n_total']:2d}  succ={d['n_success']:2d}  "
        f"n_stats={d['n_runs']:2d}  "
        f"min={d['min']:.4f}  q25={d['q25']:.4f}  med={d['median']:.4f}  "
        f"mean={d['mean']:.4f}  q75={d['q75']:.4f}  max={d['max']:.4f}"
    )

print("\n[Smooth] (stats computed only for rel_L_2_test < {:.2f})".format(SUCCESS_THRESH))
for d in smooth_rows:
    print(
        f"N={d['N']:4d}  total={d['n_total']:2d}  succ={d['n_success']:2d}  "
        f"n_stats={d['n_runs']:2d}  "
        f"min={d['min']:.4f}  q25={d['q25']:.4f}  med={d['median']:.4f}  "
        f"mean={d['mean']:.4f}  q75={d['q75']:.4f}  max={d['max']:.4f}"
    )