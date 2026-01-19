import sys
sys.path.insert(0, ".")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

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
    "axes.labelsize": 18,
    "axes.titlesize": 22,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})


def _load_pde_only(pickle_path: str):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    cfg = _to_nested_config(data["config"])
    p = build_pde_from_cfg(cfg=cfg)
    return p


def _make_grid(Dcover: np.ndarray, n=200):
    tx = np.linspace(Dcover[0, 0], Dcover[0, 1], n)
    ty = np.linspace(Dcover[1, 0], Dcover[1, 1], n)
    X, Y = np.meshgrid(tx, ty)
    t = np.vstack((X.ravel(), Y.ravel())).T
    return X, Y, t


def visualize_exact_solutions_3d(
    pkl_list,
    titles,
    out_png: str,
    grid_n: int = 200,
    Dcover: np.ndarray = None,
    zlim=None,          # e.g. (-1,1) or None
    elev: float = 25,
    azim: float = -60,
):
    assert len(pkl_list) == 3
    assert len(titles) == 3

    if Dcover is None:
        Dcover = np.array([[-1.5, 1.5], [-1.5, 1.5]])

    ps = [_load_pde_only(pkl) for pkl in pkl_list]
    for i, p in enumerate(ps):
        if p.ex_sol is None:
            raise ValueError(f"p.ex_sol is None for pkl[{i}] = {pkl_list[i]}")

    X, Y, t = _make_grid(Dcover, n=grid_n)

    # Evaluate exact solutions
    Zs = []
    for p in ps:
        Z = np.array(p.ex_sol(t)).reshape(X.shape)
        Zs.append(Z)

    # Optional shared z-limits for fair visual comparison
    if zlim is None:
        zmin = float(min(np.nanmin(Z) for Z in Zs))
        zmax = float(max(np.nanmax(Z) for Z in Zs))
        zlim = (zmin, zmax)

    fig = plt.figure(figsize=(24, 7), dpi=200)
    plt.subplots_adjust(left=0.02, right=0.98, wspace=0.10, top=0.90, bottom=0.05)

    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_surface(X, Y, Zs[i], rstride=1, cstride=1, linewidth=0, antialiased=True, cmap='viridis')

        ax.set_title(titles[i])
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$u(x)$")

        ax.set_xlim(Dcover[0, 0], Dcover[0, 1])
        ax.set_ylim(Dcover[1, 0], Dcover[1, 1])
        ax.set_zlim(zlim[0], zlim[1])

        ax.view_init(elev=elev, azim=azim)

        # Cleaner look (optional)
        ax.grid(True)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=500, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    pkl1 = "exps/exp_frac/frac_gaussain_2d_big_frac_results/frac_gaussain_2d_big_frac_seed_200.pkl"
    pkl2 = "exps/exp_frac/frac_gaussain_2d_results/frac_gaussain_2d_seed_200.pkl"
    pkl3 = "exps/exp_frac/frac_gaussain_2d_small_frac_results/frac_gaussain_2d_small_frac_seed_200.pkl"

    visualize_exact_solutions_3d(
        pkl_list=[pkl1, pkl2, pkl3],
        titles=[r"$\beta=1.5$", r"$\beta=1.0$", r"$\beta=0.5$"],
        out_png="exps/exp_frac/figs/frac_2d_3_exact_3d.png",
        grid_n=200,
        Dcover=np.array([[-1.5, 1.5], [-1.5, 1.5]]),
        zlim=None,      # shared zlim automatically computed
        elev=25,
        azim=-60,
    )