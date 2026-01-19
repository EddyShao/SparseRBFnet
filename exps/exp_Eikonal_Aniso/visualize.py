import sys
sys.path.insert(0, ".")
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import jax
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

from pde.PDE_REGISTRY import build_pde_from_cfg
from src.config.base_config import _to_nested_config






def plot_solution_2d(p, x, s, c, suppc):
    """
    Plots the forward solution.
    """
    if suppc is None:
        suppc = np.ones_like(c, dtype=bool)
    # assert p.dim == 3 

    # # Extract the domain range
    # pO = p.Omega[:-1, :]
    plt.close('all')  # Close previous figure to prevent multiple windows

    # Create a new figure
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133)

    t_x = np.linspace(p.D[0, 0], p.D[0, 1], 100)
    t_y = np.linspace(p.D[1, 0], p.D[1, 1], 100)
    X, Y = np.meshgrid(t_x, t_y)
    t = np.vstack((X.flatten(), Y.flatten())).T

    if p.ex_sol is not None:
        f1 = p.ex_sol(t).reshape(X.shape)
    # Plot exact solution
    surf1 = ax1.plot_surface(X, Y, f1, cmap='viridis', edgecolor='none')
    ax1.set_title("Exact Solution")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Compute predicted solution
    Gu = p.kernel.kappa_X_c_Xhat(x, s, c, t)
    # sigma is sigmoid of S

    # Plot predicted solution
    surf2 = ax2.plot_surface(X, Y, Gu.reshape(X.shape), cmap='viridis', edgecolor='none')
    ax2.set_title("Predicted Solution") 
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_zlabel("$f_2(x, y)$")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)


    # plot all collocation point X
    # together with error countour plot
    contour = ax3.contourf(X, Y, np.abs(Gu.reshape(100, 100) - f1), cmap='viridis')        
    # ax3.scatter(x[:, 0].flatten(), x[:, 1].flatten(), color='r', marker='x')
        # Get per-center R (shape: (N, 2, 2)); convert to numpy for matplotlib
        
    for i, (xi, yi) in enumerate(x[:, :2]):
        if suppc is not None and not bool(suppc[i]):
            continue
        # Extract the i-th R matrix
        s_i = s[i]
        r1 = p.kernel.r_min[0] + (p.kernel.r_max[0] - p.kernel.r_min[0]) * jax.nn.sigmoid(s_i[1])
        r2 = p.kernel.r_min[1] + (p.kernel.r_max[1] - p.kernel.r_min[1]) * jax.nn.sigmoid(s_i[2])
        a1, a2 = 1.0 / r1, 1.0 / r2  # Semi-major and semi-minor axes lengths

        # Rotation angle in degrees from x-axis
        angle_deg = -np.degrees(jax.nn.sigmoid(s_i[0]))  # Map to [0, 90]

        # Draw ellipse and center
        ell = patches.Ellipse((xi, yi),
                            width=2*a1, height=2*a2, angle=angle_deg,
                            edgecolor='r', facecolor='none',
                            linestyle='dashed', linewidth=1, label="Reference ellipse")
        ax3.add_patch(ell)
        ax3.scatter(xi, yi, color='r', marker='x')

    ax3.set_aspect('equal')  # Ensures circles are properly shaped
    # # set colorbars
    ax3.set_xlim(p.Omega[0, 0], p.Omega[0, 1])
    ax3.set_ylim(p.Omega[1, 0], p.Omega[1, 1])
    ax3.set_title("Collocation Points, Error Contour") 
    fig.colorbar(contour, ax=ax3, shrink=0.5, aspect=5)   

            

def plot_exp(pickle_path: str, name: str, level_id: int =2):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)


    print(data.keys())

    # Depending on how you saved, this is a dict:
    #   {"exp_result": ExperimentResult, "final_output": ..., "summary": ..., "cfg": ..., "args": ...}
    exp_result = data["experiment"]  # type: ExperimentResult

    # ----- choose level (1-based) -----
    level_id = 1
    level = exp_result.levels[level_id - 1]   # type: LevelResult

    # ----- last phase of that level -----
    last_phase = level.phases[-1]             # type: PhaseResult
    out = last_phase.output                   # type: SolverOutput

    hist = out.history

    xk_seq   = hist.get("xk", [])      # list/array over iterations
    sk_seq   = hist.get("sk", [])
    ck_seq   = hist.get("ck", [])
    supp_seq = hist.get("suppc", [])

    # If you want the *final* iterate of that phase:
    if xk_seq and sk_seq and ck_seq and supp_seq:
        xk_last   = np.array(xk_seq[-1])
        sk_last   = np.array(sk_seq[-1])
        ck_last   = np.array(ck_seq[-1])
        supp_last = np.array(supp_seq[-1])
    else:
        xk_last = sk_last = ck_last = supp_last = None

    from pde.PDE_REGISTRY import build_pde_from_cfg
    from src.config.base_config import _to_nested_config
    cfg = _to_nested_config(data['config'])
    p = build_pde_from_cfg(cfg=cfg)

    plt.clf()
    plot_solution_2d(p, xk_last, sk_last, ck_last, supp_last)
    os.makedirs("exps/exp_Eikonal_Aniso/figs/", exist_ok=True)
    plt.savefig(f"exps/exp_Eikonal_Aniso/figs/{name}.png")
    plt.show()

if __name__ == "__main__":
    pickle_path_list = [f"exps/exp_Eikonal_Aniso/eikonal_2d_aniso_new_results/eikonal_2d_aniso_new_seed_{seed}.pkl" for seed in range(200, 219)]
    for pickle_path in pickle_path_list:
        name = pickle_path.split("/")[-1].replace(".pkl", "")
        plot_exp(pickle_path, name)