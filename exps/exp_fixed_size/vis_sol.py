import json
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load JSON from a file
# -----------------------
json_path = "exps/exp_fixed_size/aggregated_results.json"
with open(json_path, "r") as f:
    data = json.load(f)

# -----------------------
# Extract per-N stats (exclude adaptive)
# -----------------------
rows = []
for key, block in data.items():
    if key == "second_order_full":
        continue
    if not key.startswith("N_"):
        continue

    N = int(key.split("_")[1])
    vals = np.array([r["rel_L_2_test"] for r in block["runs"]], dtype=float)

    rows.append({
        "N": N,
        "min": float(np.min(vals)),
        "q25": float(np.quantile(vals, 0.25)),
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "q75": float(np.quantile(vals, 0.75)),
        "max": float(np.max(vals)),
        "n_runs": int(len(vals)),
    })

rows = sorted(rows, key=lambda d: d["N"])

N_list = np.array([d["N"] for d in rows], dtype=int)
vmin   = np.array([d["min"] for d in rows], dtype=float)
vq25   = np.array([d["q25"] for d in rows], dtype=float)
vmed   = np.array([d["median"] for d in rows], dtype=float)
vmean  = np.array([d["mean"] for d in rows], dtype=float)
vq75   = np.array([d["q75"] for d in rows], dtype=float)
vmax   = np.array([d["max"] for d in rows], dtype=float)


bench_block = data.get("second_order_full", None)
bench_vals = None
bench_mean = None
bench_median = None

if bench_block is not None:
    bench_vals = np.array([r["rel_L_2_test"] for r in bench_block["runs"]], dtype=float)
    bench_mean = float(np.mean(bench_vals))
    bench_median = float(np.median(bench_vals))
    bench_std = float(np.std(bench_vals))
    bench_q25 = float(np.quantile(bench_vals, 0.25))
    bench_q75 = float(np.quantile(bench_vals, 0.75))

# -----------------------
# Plot: range + IQR (no fill) + mean/median vs N
# -----------------------
# -----------------------
# Plot: range (I-shape) + IQR (bar) + mean/median vs N
# -----------------------
fig, ax = plt.subplots(figsize=(8.5, 4.8))

# Widths (multiplicative, symmetric in log-x)
cap_factor = 1.06   # for I-shape caps
bar_factor = 1.03   # for IQR bar width

for x, lo, q1, med, mean, q3, hi in zip(
    N_list, vmin, vq25, vmed, vmean, vq75, vmax
):
    # ---- Range: italic "I" (min–max with caps) ----
    ax.vlines(x, lo, hi, linewidth=1.8, alpha=0.8)
    ax.hlines(lo, x / cap_factor, x * cap_factor, linewidth=1.8)
    ax.hlines(hi, x / cap_factor, x * cap_factor, linewidth=1.8)

    # ---- IQR: thick vertical bar ----
    ax.vlines(x, q1, q3, linewidth=6, alpha=0.8)

ax.axhspan(
        bench_mean - bench_std,
        bench_mean + bench_std,
        alpha=0.18,
        zorder=0,
        label="Adaptive (mean ± std)",
        color="red"
    )

# mean line
ax.axhline(
    bench_mean,
    linestyle="--",
    linewidth=2.2,
    zorder=1,
    label="Adaptive mean",
    color="red"
)

# ---- Median and mean ----
ax.plot(N_list, vmed, marker="o", linewidth=2, label="Median")
# ax.plot(N_list, vmean, marker="s", linewidth=2, label="Mean")

# ---- Axes ----
ax.set_xscale("log", base=2)
ax.set_yscale("log")

ax.set_xticks(N_list)
ax.get_xaxis().set_major_formatter(lambda x, pos: f"{int(x)}")

ax.set_xlabel("N")
ax.set_ylabel(r"$\mathrm{err}^{\mathrm{rel}}_{2}$")
ax.set_title(r"Error distribution vs N")

ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.5)
ax.legend(frameon=True)

plt.tight_layout()
plt.savefig("exps/exp_fixed_size/fixed_size_error_vs_N.png", dpi=300)
plt.show()

# -----------------------
# Print extracted table (optional)
# -----------------------
for d in rows:
    print(
        f"N={d['N']:4d}  runs={d['n_runs']:2d}  "
        f"min={d['min']:.4f}  q25={d['q25']:.4f}  med={d['median']:.4f}  "
        f"mean={d['mean']:.4f}  q75={d['q75']:.4f}  max={d['max']:.4f}"
    )