import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------
# Matplotlib style
# -----------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,

    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

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
def parse_N(key):
    try:
        return int(key.split("_")[1])
    except Exception:
        return None

def is_fixed(key):
    return key.startswith("N_") and not key.endswith("_smooth") and not key.endswith("_smooths")

def is_smooth(key):
    return key.startswith("N_") and (key.endswith("_smooth") or key.endswith("_smooths"))

# -----------------------
# Collect runtimes
# -----------------------
fixed_rows = []
smooth_rows = []

for key, block in data.items():
    if key == "second_order_full":
        continue
    if not key.startswith("N_"):
        continue
    if "runtime" not in block:
        continue

    N = parse_N(key)
    if N is None:
        continue

    t_mean = float(block["runtime"]["mean"])

    if is_fixed(key):
        fixed_rows.append((N, t_mean))
    elif is_smooth(key):
        smooth_rows.append((N, t_mean))

fixed_rows.sort(key=lambda x: x[0])
smooth_rows.sort(key=lambda x: x[0])

N_fixed, t_fixed = zip(*fixed_rows) if fixed_rows else ([], [])
N_smooth, t_smooth = zip(*smooth_rows) if smooth_rows else ([], [])

N_fixed = np.array(N_fixed, dtype=int)
t_fixed = np.array(t_fixed, dtype=float)

N_smooth = np.array(N_smooth, dtype=int)
t_smooth = np.array(t_smooth, dtype=float)

# -----------------------
# Adaptive benchmark runtime (mean only)
# -----------------------
bench_block = data.get("second_order_full", None)
bench_mean = None
if bench_block is not None and "runtime" in bench_block:
    bench_mean = float(bench_block["runtime"]["mean"])

# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(figsize=(10, 6.4))

# Fixed-size network
ax.plot(
    N_fixed, t_fixed,
    marker="o",
    label="Fixed-size network + $\ell^{{1}}$ regularization (mean)",
    c="tab:blue",
)

# Smooth network
ax.plot(
    N_smooth, t_smooth,
    marker="^",
    linestyle="-.",
    label="Fixed-size network + $\ell^{2}$ regularization (mean)",
    c="tab:orange",
)

# Adaptive benchmark
if bench_mean is not None:
    ax.axhline(
        bench_mean,
        linestyle="--",
        linewidth=2.2,
        label="Our method: adaptive (mean)",
        color="red",
    )

# Axes
ax.set_xlim(min(min(N_fixed, default=1), min(N_smooth, default=1)) * 0.8,
            max(max(N_fixed, default=1), max(N_smooth, default=1)) * 1.35)
ax.set_xscale("log", base=2)
ax.set_yscale("log")

xticks = sorted(set(N_fixed.tolist() + N_smooth.tolist()))
ax.set_xticks(xticks)
ax.get_xaxis().set_major_formatter(lambda x, pos: f"{int(x)}")

ax.set_xlabel(r"$N_{\mathrm{fixed}}$")
ax.set_ylabel("Runtime (s)")

ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.5)
ax.legend(frameon=True)

plt.tight_layout()
plt.savefig("exps/exp_fixed_size/fixed_size_runtime_vs_N.png", dpi=300)
plt.show()