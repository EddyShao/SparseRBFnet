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
    # Font: portable on HPC
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],   # ships with matplotlib
    "mathtext.fontset": "cm",        # Computer Modern math (or use "stix" if you prefer)
    "axes.unicode_minus": False,

    # sizes
    "font.size": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
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


def _make_grid(p, n=200):
    tx = np.linspace(p.D[0, 0], p.D[0, 1], n)
    ty = np.linspace(p.D[1, 0], p.D[1, 1], n)
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
    """x: (N,2) -> boolean mask for centers inside [-box, box]^2."""
    return (np.abs(x[:, 0]) <= box) & (np.abs(x[:, 1]) <= box)


def _overlay_shapes(ax, p, x, s, suppc, vis_box=1.2):
    """
    Overlays circles (iso) or ellipses (aniso) for centers inside [-vis_box, vis_box]^2.
    """
    N = x.shape[0]
    if suppc is None:
        suppc = np.ones((N,), dtype=bool)
    else:
        suppc = np.array(suppc, dtype=bool).reshape(-1)

    inside = _in_box_mask(x[:, :2], box=vis_box)
    keep = suppc & inside

    is_aniso = bool(getattr(p.kernel, "anisotropic", False))

    if not is_aniso:
        sigma = np.array(p.kernel.sigma(s)).reshape(-1)  # (N,)

        for (xi, yi, r, ok) in zip(x[:, 0], x[:, 1], sigma, keep):
            if not ok:
                continue
            ax.scatter(xi, yi, c="black", s=300, marker="*", zorder=5)
            circ = plt.Circle(
                (xi, yi), float(r),
                edgecolor="r", facecolor="none",
                linestyle="dotted", linewidth=2,
                zorder=4
            )
            ax.add_patch(circ)

    else:
        for i, (xi, yi) in enumerate(x[:, :2]):
            if not keep[i]:
                continue

            s_i = s[i]

            r1 = p.kernel.r_min[0] + (p.kernel.r_max[0] - p.kernel.r_min[0]) * jax.nn.sigmoid(s_i[1])
            r2 = p.kernel.r_min[1] + (p.kernel.r_max[1] - p.kernel.r_min[1]) * jax.nn.sigmoid(s_i[2])

            a1, a2 = 1.0 / float(r1), 1.0 / float(r2)

            # angle in [0, pi] radians, then degrees
            angle_deg = -np.degrees(np.pi * float(jax.nn.sigmoid(s_i[0])))

            ax.scatter(xi, yi, c="black", s=400, marker="*", zorder=5)
            ell = patches.Ellipse(
                (xi, yi),
                width=2 * a1, height=2 * a2,
                angle=angle_deg,
                edgecolor="r", facecolor="none",
                linestyle="dotted", linewidth=3.0,
                zorder=4
            )
            ax.add_patch(ell)


def compare_error_contours_for_paper(
    iso_pkl: str,
    aniso_pkl: str,
    out_png: str,
    level_id: int = 1,
    grid_n: int = 200,
    vis_xlim: float = 1.35,
    vis_box_for_kernels: float = 1.2,
    levels: int = 200,
    cmap: str = "coolwarm",
    use_robust_clim: bool = False,
    robust_q: float = 0.995,
):
    p_iso,  x_iso,  s_iso,  c_iso,  supp_iso  = _load_final_iterate(iso_pkl,  level_id=level_id)
    p_aniso, x_aniso, s_aniso, c_aniso, supp_aniso = _load_final_iterate(aniso_pkl, level_id=level_id)

    # grid (assumes same domain; if not, you’ll still get a plot but domain mismatch is on you)
    X, Y, t = _make_grid(p_iso, n=grid_n)

    err_iso   = _compute_error_on_grid(p_iso,   x_iso,   s_iso,   c_iso,   t, X.shape)
    err_aniso = _compute_error_on_grid(p_aniso, x_aniso, s_aniso, c_aniso, t, X.shape)

    # shared color limits
    if use_robust_clim:
        vmax = float(np.quantile(np.r_[err_iso.ravel(), err_aniso.ravel()], robust_q))
    else:
        vmax = float(max(np.nanmax(err_iso), np.nanmax(err_aniso)))
    vmin = 0.0

    # shared normalization + shared levels
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    level_vals = np.linspace(vmin, vmax, levels)

    # figure layout (avoid constrained_layout because we add custom colorbar axes)
    fig, axes = plt.subplots(1, 2, figsize=(19, 13), dpi=200, gridspec_kw={"wspace": 0.01})
    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.15)

    # ISO panel
    axes[0].contourf(X, Y, err_iso, levels=level_vals, cmap=cmap, norm=norm, extend="max")
    _overlay_shapes(axes[0], p_iso, x_iso, s_iso, supp_iso, vis_box=vis_box_for_kernels)
    axes[0].set_title("Isotropic kernel (#Kernels = {})".format(np.sum(supp_iso)))
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlim(-vis_xlim, vis_xlim)
    axes[0].set_ylim(-vis_xlim, vis_xlim)

    # ANISO panel
    axes[1].contourf(X, Y, err_aniso, levels=level_vals, cmap=cmap, norm=norm, extend="max")
    _overlay_shapes(axes[1], p_aniso, x_aniso, s_aniso, supp_aniso, vis_box=vis_box_for_kernels)
    axes[1].set_title("Anisotropic kernel (#Kernels = {})".format(np.sum(supp_aniso)))
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlim(-vis_xlim, vis_xlim)
    axes[1].set_ylim(-vis_xlim, vis_xlim)

    for ax in axes:
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.tick_params(labelsize=24)

    # bottom shared colorbar built from a ScalarMappable (always consistent)
    formatter = mticker.ScalarFormatter(useMathText=True)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.225, 0.06, 0.55, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", format=formatter)
    cbar.set_label("Absolute error", fontsize=18)
    cbar.ax.tick_params(labelsize=12)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    iso_pkl   = "exps/exp_Eikonal_Aniso/eikonal_2d_iso_new_results/eikonal_2d_iso_new_seed_205.pkl"
    aniso_pkl = "exps/exp_Eikonal_Aniso/eikonal_2d_aniso_new_results/eikonal_2d_aniso_new_seed_205.pkl"

    compare_error_contours_for_paper(
        iso_pkl=iso_pkl,
        aniso_pkl=aniso_pkl,
        out_png="exps/exp_Eikonal_Aniso/figs/aniso_comparison.png",
        grid_n=200,
        vis_xlim=1.5,
        vis_box_for_kernels=1.5,   # <-- excludes kernel centers outside [-1.2,1.2]^2
        levels=200,
        cmap="coolwarm",
        use_robust_clim=False,
        robust_q=0.995,
    )