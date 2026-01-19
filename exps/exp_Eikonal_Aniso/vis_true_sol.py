import sys
sys.path.insert(0, ".")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pde.PDE_REGISTRY import build_pde_from_cfg
from src.config.base_config import _to_nested_config


# =========================
# Paper-style rcParams (HPC-safe)
# =========================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,

    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 26,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})


def _build_pde_from_pickle(pickle_path: str):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    cfg = _to_nested_config(data["config"])
    return build_pde_from_cfg(cfg=cfg)


def _make_grid(p, n=200):
    tx = np.linspace(p.D[0, 0], p.D[0, 1], n)
    ty = np.linspace(p.D[1, 0], p.D[1, 1], n)
    X, Y = np.meshgrid(tx, ty)
    t = np.vstack((X.ravel(), Y.ravel())).T
    return X, Y, t


def plot_true_solution_3d(
    pickle_path: str,
    out_png: str,
    grid_n: int = 200,
    cmap: str = "coolwarm",
    elev: float = 30,
    azim: float = -60,
):
    p = _build_pde_from_pickle(pickle_path)

    if p.ex_sol is None:
        raise ValueError("p.ex_sol is None — cannot plot true solution.")

    # Grid + evaluation
    X, Y, t = _make_grid(p, n=grid_n)
    U = np.array(p.ex_sol(t)).reshape(X.shape)

    # Figure
    fig = plt.figure(figsize=(12, 12), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.12)

    # Surface
    surf = ax.plot_surface(
        X, Y, U,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )

    # View / labels
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$u(x)$", labelpad=10)

    ax.set_xlim(p.D[0, 0], p.D[0, 1])
    ax.set_ylim(p.D[1, 0], p.D[1, 1])

    # Cleaner z ticks
    ax.zaxis.set_major_locator(mticker.MaxNLocator(5))

    # # Colorbar (bottom, paper-style)
    # formatter = mticker.ScalarFormatter(useMathText=True)
    # cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.03])
    # cbar = fig.colorbar(surf, cax=cbar_ax, orientation="horizontal", format=formatter)
    # cbar.set_label("True solution value", fontsize=16)
    # cbar.ax.tick_params(labelsize=12)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    pkl = "exps/exp_Eikonal_Aniso/eikonal_2d_iso_results/eikonal_2d_iso_seed_207.pkl"

    plot_true_solution_3d(
        pickle_path=pkl,
        out_png="exps/exp_Eikonal_Aniso/figs/true_solution_3d.png",
        grid_n=200,
        cmap="viridis",
        elev=30,
        azim=-60,
    )