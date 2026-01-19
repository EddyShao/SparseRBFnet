import sys
sys.path.insert(0, ".")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import matplotlib as mpl

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

from pde.PDE_REGISTRY import build_pde_from_cfg
from src.config.base_config import _to_nested_config


# =========================
# Paper-style rcParams
# =========================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
})


def _load_final_iterate(pickle_path: str, level_id: int = 1):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    exp_result = data["experiment"]
    level = exp_result.levels[level_id - 1]
    last_phase = level.phases[-1]
    out = last_phase.output
    hist = out.history

    xk_seq   = hist.get("xk", [])
    sk_seq   = hist.get("sk", [])
    ck_seq   = hist.get("ck", [])
    supp_seq = hist.get("suppc", [])

    if not (xk_seq and sk_seq and ck_seq):
        raise ValueError(f"Missing xk/sk/ck history in {pickle_path}")

    x_last = np.array(xk_seq[-1])
    s_last = np.array(sk_seq[-1])
    c_last = np.array(ck_seq[-1])
    supp_last = np.array(supp_seq[-1]) if supp_seq else None

    cfg = _to_nested_config(data["config"])
    p = build_pde_from_cfg(cfg=cfg)

    return p, x_last, s_last, c_last, supp_last


def _make_grid_from_domain(D: np.ndarray, n=200):
    """
    D: (d,2) domain bounds
    Assumes 2D for contour plotting.
    """
    tx = np.linspace(D[0, 0], D[0, 1], n)
    ty = np.linspace(D[1, 0], D[1, 1], n)
    X, Y = np.meshgrid(tx, ty)
    t = np.vstack((X.ravel(), Y.ravel())).T
    return X, Y, t


def _compute_error_on_grid(p, x, s, c, t, Xshape):
    if p.ex_sol is None:
        raise ValueError("p.ex_sol is None; cannot compute error contour.")
    f_ex = np.array(p.ex_sol(t)).reshape(Xshape)
    f_pr = np.array(p.kernel.kappa_X_c_Xhat(x, s, c, t)).reshape(Xshape)
    return np.abs(f_pr - f_ex)


def _in_box_mask(x, box=1.2):
    return (np.abs(x[:, 0]) <= box) & (np.abs(x[:, 1]) <= box)


def _overlay_iso_shapes(ax, p, x, s, suppc, vis_box=1.2):
    """
    Isotropic only: overlay circles.
    """
    N = x.shape[0]
    if suppc is None:
        suppc = np.ones((N,), dtype=bool)
    else:
        suppc = np.array(suppc, dtype=bool).reshape(-1)

    inside = _in_box_mask(x[:, :2], box=vis_box)
    keep = suppc & inside

    sigma = np.array(p.kernel.sigma(s)).reshape(-1)  # (N,)

    for (xi, yi, r, ok) in zip(x[:, 0], x[:, 1], sigma, keep):
        if not ok:
            continue
        ax.scatter(xi, yi, c="black", s=220, marker="*", zorder=5)
        circ = plt.Circle(
            (xi, yi), float(r),
            edgecolor="r", facecolor="none",
            linestyle="dotted", linewidth=2,
            zorder=4
        )
        ax.add_patch(circ)


def compare_error_contours_3_iso(
    pkl_list,                 # list[str] length=3
    out_png: str,
    titles=None,              # list[str] length=3
    level_id: int = 1,
    grid_n: int = 200,
    vis_xlim: float = 1.5,
    vis_box_for_kernels: float = 1.2,
    levels: int = 200,
    cmap: str = "coolwarm",
    use_robust_clim: bool = False,
    robust_q: float = 0.995,
):
    assert len(pkl_list) == 3, "pkl_list must have length 3"
    if titles is None:
        titles = [f"Run {i+1}" for i in range(3)]
    assert len(titles) == 3, "titles must have length 3"

    # Load all 3
    payloads = []
    for pkl in pkl_list:
        p, x, s, c, supp = _load_final_iterate(pkl, level_id=level_id)
        payloads.append((p, x, s, c, supp))

    # Use a shared grid: safest is to pick a "covering domain" of all three p.D
    # (assumes all are 2D). If domains differ, this still works.
    Ds = [np.array(p.D) for (p, *_rest) in payloads]
    Dmin = np.min(np.stack([D[:, 0] for D in Ds], axis=0), axis=0)
    Dmax = np.max(np.stack([D[:, 1] for D in Ds], axis=0), axis=0)
    # Dcover = np.stack([Dmin, Dmax], axis=1)  # shape (2,2)
    Dcover = np.array([
        [-1.5, 1.5],
        [-1.5, 1.5]
    ])

    X, Y, t = _make_grid_from_domain(Dcover, n=grid_n)

    # Compute errors
    errs = []
    for (p, x, s, c, _supp) in payloads:
        err = _compute_error_on_grid(p, x, s, c, t, X.shape)
        errs.append(err)

    # Shared color limits across all 3
    all_vals = np.concatenate([e.ravel() for e in errs])
    if use_robust_clim:
        vmax = float(np.quantile(all_vals, robust_q))
    else:
        vmax = float(np.nanmax(all_vals))
    vmin = 0.0

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    level_vals = np.linspace(vmin, vmax, levels)

    # Figure: 1x3
    fig, axes = plt.subplots(1, 3, figsize=(27, 12), dpi=500, gridspec_kw={"wspace": 0.02})
    plt.subplots_adjust(left=0.02, right=0.985, top=0.88, bottom=0.16)

    for i, ax in enumerate(axes):
        p, x, s, c, supp = payloads[i]
        err = errs[i]

        ax.contourf(X, Y, err, levels=level_vals, cmap=cmap, norm=norm, extend="max")
        # isotropic overlay only
        _overlay_iso_shapes(ax, p, x, s, supp, vis_box=vis_box_for_kernels)

        nk = int(np.sum(np.array(supp, dtype=bool))) if supp is not None else int(x.shape[0])
        ax.set_title(f"{titles[i]} (#Kernels = {nk})")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-vis_xlim, vis_xlim)
        ax.set_ylim(-vis_xlim, vis_xlim)

        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlabel(r"$x_1$")
        if i == 0:
            ax.set_ylabel(r"$x_2$")
        else:
            ax.set_ylabel("")

    # Bottom shared colorbar
    formatter = mticker.ScalarFormatter(useMathText=True)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.24, 0.07, 0.52, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", format=formatter)
    cbar.set_label("Absolute error", fontsize=18)
    cbar.ax.tick_params(labelsize=12)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    pkl1 = "exps/exp_frac/frac_gaussain_2d_big_frac_results/frac_gaussain_2d_big_frac_seed_200.pkl"
    pkl2 = "exps/exp_frac/frac_gaussain_2d_results/frac_gaussain_2d_seed_200.pkl"
    pkl3 = "exps/exp_frac/frac_gaussain_2d_small_frac_results/frac_gaussain_2d_small_frac_seed_200.pkl"

    compare_error_contours_3_iso(
        pkl_list=[pkl1, pkl2, pkl3],
        titles=[r"$\beta=1.5$", r"$\beta=1.0$", r"$\beta=0.5$"],
        out_png="exps/exp_frac/figs/frac_2d_3_error.png",
        level_id=1,
        grid_n=200,
        vis_xlim=1.5,
        vis_box_for_kernels=1.5,
        levels=200,
        cmap="coolwarm",
        use_robust_clim=False,
        robust_q=0.995,
    )