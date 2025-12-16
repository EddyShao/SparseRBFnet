# scripts/solve_pde_eps_homotopy.py

import sys
sys.path.append("./")

import argparse
import os
import pickle
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np  # optional

from src.config.base_config import load_config_from_args
from src.config.exp_config import (
    SolverGlobalConfig,
    SolverPhaseConfig,
    HomotopyConfig,
)
from src.config.output_config import (
    SolverOutput,
    PhaseResult,
    LevelResult,
    ExperimentResult,
)
from src.solver.SOLVER_REGISTRY import SOLVER_REGISTRY
from src.utils import Objective, compute_errors, compute_rhs
from pde.PDE_REGISTRY import build_pde_from_cfg


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="PDE solver with epsilon homotopy.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=200,
        help="Random seed, override config (if given)."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override save_dir from YAML (optional).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Solver utilities
# ----------------------------------------------------------------------

def build_alg_opts(global_cfg: SolverGlobalConfig,
                   phase_cfg: SolverPhaseConfig,
                   cfg) -> dict:
    """Build flat dict of options expected by solvers."""
    opts = {
        "alpha": global_cfg.alpha,
        "T": global_cfg.T,
        "gamma": global_cfg.gamma,
        "Ntrial": global_cfg.Ntrial,
        "blocksize": global_cfg.blocksize,
        "print_every": global_cfg.print_every,
        "plot_every": global_cfg.plot_every,
        "max_step": phase_cfg.max_step,
        "TOL": phase_cfg.TOL,
        "plot_final": phase_cfg.plot_final,
        # PDE-related options that some solvers expect:
        "anisotropic": bool(cfg.kernel.anisotropic),
        "scale": float(cfg.pde.scale),
        "sampling": str(cfg.pde.sampling),
        "seed": int(cfg.pde.seed),
    }
    return opts


def run_phase(
    p,
    rhs,
    global_cfg: SolverGlobalConfig,
    phase_cfg: SolverPhaseConfig,
    cfg,
) -> SolverOutput:
    """
    Run a single phase (e.g. warmup_outer or refine).

    - Builds flat alg_opts from global + phase + PDE config.
    - Calls the registered solver.
    - Wraps the raw dict into SolverOutput.
    - Sets p.u_zero from final active coefficients as warm start.
    """
    solver_fn = SOLVER_REGISTRY[phase_cfg.solver_name]
    alg_opts = build_alg_opts(global_cfg, phase_cfg, cfg)

    print(f"\n=== Phase: {phase_cfg.name} (solver={phase_cfg.solver_name}) ===")
    print(
        f"alpha={global_cfg.alpha:.3e}, T={global_cfg.T:.3e}, "
        f"max_step={phase_cfg.max_step}, TOL={phase_cfg.TOL:.1e}"
    )

    raw_alg_out = solver_fn(p, rhs, alg_opts)
    alg_out = SolverOutput.from_raw(raw_alg_out)

    # Warm start: use *masked* final solution as p.u_zero for next phase/level
    x_last = alg_out.final["x"]
    s_last = alg_out.final["s"]
    c_last = alg_out.final["u"]

    if x_last is None or s_last is None or c_last is None or len(c_last) == 0:
        # fall back to zeros if solver returned no active support
        pad_size = getattr(p, "init_pad_size", 16)
        p.u_zero = {
            "x": jnp.zeros((pad_size, p.d)),
            "s": jnp.zeros((pad_size, p.dim - p.d)),
            "u": jnp.zeros((pad_size,)),
        }
    else:
        p.u_zero = {
            "x": jnp.array(x_last),
            "s": jnp.array(s_last),
            "u": jnp.array(c_last),
        }

    return alg_out


def run_all_phases_one_level(
    p,
    rhs,
    global_cfg: SolverGlobalConfig,
    phase_cfgs,
    cfg,
    level_index: int,
) -> LevelResult:
    """
    Run [phase_1, phase_2, ...] once for fixed (alpha, T, epsilon),
    and wrap results into a LevelResult.
    """
    phase_results: list[PhaseResult] = []

    for phase_cfg in phase_cfgs:
        alg_out = run_phase(p, rhs, global_cfg, phase_cfg, cfg)
        phase_results.append(
            PhaseResult(
                name=phase_cfg.name,
                solver_name=phase_cfg.solver_name,
                output=alg_out,
            )
        )

    level_res = LevelResult(
        level_index=level_index,
        alpha=global_cfg.alpha,
        T=global_cfg.T,
        phases=phase_results,
    )
    # attach epsilon just for convenience
    if hasattr(p, "epsilon"):
        level_res.epsilon = float(p.epsilon)
    return level_res


# ----------------------------------------------------------------------
# RHS builder for current epsilon
# ----------------------------------------------------------------------

def build_rhs_for_current_epsilon(p, cfg):
    """
    Build training RHS for the *current* value of p.epsilon.
    Mirrors the logic in your original main().
    """
    rhs = p.f(p.xhat)
    if getattr(cfg.pde, "add_noise", False):
        rhs_mag = jnp.max(jnp.abs(rhs[:-p.Nx_bnd]))
        noise = jnp.random.randn(p.Nx) * 0.01 * rhs_mag
        rhs = rhs + noise
    rhs = rhs.at[-p.Nx_bnd:].set(p.ex_sol(p.xhat_bnd))
    return rhs


# ----------------------------------------------------------------------
# Homotopy driver (epsilon + possibly alpha/T)
# ----------------------------------------------------------------------

def _get_initial_epsilon(p, cfg, hom_cfg: HomotopyConfig):
    """
    Decide the starting epsilon for homotopy.

    Priority:
      1) hom_cfg.eps0 if eps_enabled
      2) p.epsilon (if exists)
      3) cfg.pde.epsilon or cfg.pde.data["epsilon"]
    """
    if getattr(hom_cfg, "eps_enabled", False) and getattr(hom_cfg, "eps0", None) is not None:
        return hom_cfg.eps0

    if hasattr(p, "epsilon"):
        return p.epsilon

    if hasattr(cfg.pde, "epsilon"):
        return cfg.pde.epsilon

    # if you store it in .data dict:
    if hasattr(cfg.pde, "data") and "epsilon" in cfg.pde.data:
        return cfg.pde.data["epsilon"]

    return None


def _set_epsilon_everywhere(p, cfg, epsilon: float):
    """
    Single place where we update epsilon in:
      - p.epsilon
      - p.kernel.epsilon
      - cfg.pde (and cfg.pde.data if used)
    """
    if epsilon is None:
        return

    # PDE object
    p.epsilon = float(epsilon)

    # kernel inside PDE
    if hasattr(p, "kernel"):
        p.kernel.epsilon = float(epsilon)

    # config: keep pde epsilon in sync
    if hasattr(cfg.pde, "data"):
        cfg.pde.data["epsilon"] = float(epsilon)
    else:
        setattr(cfg.pde, "epsilon", float(epsilon))


def run_solver_schedule(p, cfg) -> ExperimentResult:
    """
    Top-level driver with homotopy:

    - Reads SolverGlobalConfig and HomotopyConfig.
    - Supports alpha/T homotopy as before.
    - Additionally supports epsilon homotopy via:
        homotopy.eps_enabled, homotopy.eps0, homotopy.eps_factor

    At each level:
      - set epsilon via p.epsilon & p.kernel.epsilon & cfg.pde.epsilon
      - rebuild RHS for that epsilon,
      - run all phases,
      - warm-start next level via p.u_zero.
    """
    global_cfg = SolverGlobalConfig.from_cfg(cfg)
    hom_cfg = HomotopyConfig.from_cfg(cfg)
    phase_cfgs = [SolverPhaseConfig.from_cfg(ph) for ph in cfg.solver.phases]

    level_results: list[LevelResult] = []
    n_levels = hom_cfg.num_levels if hom_cfg.enabled else 1

    # Initialize epsilon
    current_epsilon = _get_initial_epsilon(p, cfg, hom_cfg)

    for level in range(n_levels):
        print(f"\n######## Homotopy level {level + 1}/{n_levels} ########")

        # --- update epsilon on PDE object + kernel + config ---
        if getattr(hom_cfg, "eps_enabled", False) and current_epsilon is not None:
            _set_epsilon_everywhere(p, cfg, current_epsilon)
            print(f"epsilon = {current_epsilon:.4e}")
        else:
            if current_epsilon is not None:
                _set_epsilon_everywhere(p, cfg, current_epsilon)
                print(f"epsilon (no homotopy) = {current_epsilon:.4e}")

        print(f"alpha = {global_cfg.alpha:.3e}, T = {global_cfg.T:.3e}")

        # --- build RHS for this epsilon ---
        rhs = build_rhs_for_current_epsilon(p, cfg)

        # --- run all phases for this level ---
        level_res = run_all_phases_one_level(
            p, rhs, global_cfg, phase_cfgs, cfg, level_index=level
        )
        # ensure epsilon is recorded
        if hasattr(p, "epsilon"):
            level_res.epsilon = float(p.epsilon)

        level_results.append(level_res)

        # --- update homotopy parameters for next level ---
        if getattr(hom_cfg, "eps_enabled", False) and current_epsilon is not None:
            current_epsilon *= hom_cfg.eps_factor

        if hom_cfg.enabled:
            global_cfg.alpha *= hom_cfg.alpha_factor
            global_cfg.T *= hom_cfg.T_factor

    return ExperimentResult(levels=level_results)


# ----------------------------------------------------------------------
# Evaluation & saving
# ----------------------------------------------------------------------

def evaluate_solution(
    p,
    solver_global: SolverGlobalConfig,
    alg_out: SolverOutput,
    level_index: int,
    alpha: float,
    epsilon: Optional[float],
    cfg,
    *,
    Ntest: int = 20,
):
    """
    Compute train/test errors and residuals for the final solution at a given
    homotopy level (alpha, epsilon).

    - Sets epsilon in p & kernel before evaluating.
    - Regenerates a test set on the fly.
    - Rebuilds train RHS for the *current* epsilon for residuals.
    """
    # ensure epsilon is consistent at evaluation
    _set_epsilon_everywhere(p, cfg, epsilon)

    # -----------------------------
    # 1) Regenerate test set
    # -----------------------------
    Ntest_int, Ntest_bnd = (Ntest - 2) ** p.d, 2 * p.d * (Ntest - 2) ** (p.d - 1)
    p.test_int, p.test_bnd = p.sample_obs(
        Ntest_int,
        Ntest_bnd,
        method="grid",
    )

    rhs_test_int = p.f(p.test_int)
    rhs_test_bnd = p.ex_sol(p.test_bnd)
    rhs_test = jnp.concatenate((rhs_test_int, rhs_test_bnd))

    # Objective for test set
    p.obj_test = Objective(
        p.test_int.shape[0],
        p.test_bnd.shape[0],
        scale=p.scale if hasattr(p, "scale") else 0.0,
    )

    # -----------------------------
    # 2) Final solution from SolverOutput
    # -----------------------------
    xk_final = alg_out.final["x"]
    sk_final = alg_out.final["s"]
    ck_final = alg_out.final["u"]

    # Test errors
    errors_test = compute_errors(
        p, xk_final, sk_final, ck_final, p.test_int, p.test_bnd
    )
    errors_test = {k + "_test": v for k, v in errors_test.items()}

    # Train errors
    errors_train = compute_errors(
        p, xk_final, sk_final, ck_final, p.xhat_int, p.xhat_bnd
    )
    errors_train = {k + "_train": v for k, v in errors_train.items()}

    # -----------------------------
    # 3) Residuals
    # -----------------------------
    rhs_train = p.f(p.xhat)
    rhs_train = rhs_train.at[-p.Nx_bnd:].set(p.ex_sol(p.xhat_bnd))

    yk_train, _, _ = compute_rhs(
        p, xk_final, sk_final, ck_final, p.xhat_int, p.xhat_bnd
    )
    misfit_train = yk_train - rhs_train
    residue_train = p.obj.F(misfit_train)

    yk_test, _, _ = compute_rhs(
        p, xk_final, sk_final, ck_final, p.test_int, p.test_bnd
    )
    misfit_test = yk_test - rhs_test
    residue_test = p.obj_test.F(misfit_test)

    # -----------------------------
    # 4) Logging
    # -----------------------------
    print()
    print("#" * 20)
    if epsilon is not None:
        print(f"level {level_index}: alpha ~ {alpha:.1e}, epsilon ~ {epsilon:.3e}")
    else:
        print(f"level {level_index}: alpha ~ {alpha:.1e}")
    print(
        "L_inf error test (boundary): {L_inf_bnd_test:.2e}\n"
        "L_inf error test (interior): {L_inf_int_test:.2e}\n"
        "L_inf error test (total): {L_inf_test:.2e}\n"
        "L_2 error test: {L_2_test:.2e}\n"
        "rel L_2 error test: {rel_L_2_test:.2e}"
        .format(**errors_test)
    )
    print(f"residue test: {residue_test:.2e}")
    print(
        "L_inf error train (boundary): {L_inf_bnd_train:.2e}\n"
        "L_inf error train (interior): {L_inf_int_train:.2e}\n"
        "L_inf error train (total): {L_inf_train:.2e}\n"
        "L_2 error train: {L_2_train:.2e}\n"
        "rel L_2 error train: {rel_L_2_train:.2e}"
        .format(**errors_train)
    )
    print(f"residue train: {residue_train:.2e}")
    print(f"final support: {alg_out.meta['final_supp']}")
    print("#" * 20)
    print()

    summary = {
        "alpha": float(alpha),
        "epsilon": float(epsilon) if epsilon is not None else None,
        "residue_test": float(residue_test),
        "residue_train": float(residue_train),
        "final_supp": int(alg_out.meta["final_supp"]),
    }
    summary.update({k: float(v) for k, v in errors_test.items()})
    summary.update({k: float(v) for k, v in errors_train.items()})
    return summary


def save_results_pickle(
    exp_result: Optional[ExperimentResult],
    final_output: SolverOutput,
    summary: dict,
    cfg,
    args,
) -> None:
    """
    Save full experiment result + final solver output + summary + config using pickle.
    """
    save_dir = args.save_dir
    if save_dir is None:
        print("No save_dir specified; skipping saving.")
        return

    os.makedirs(save_dir, exist_ok=True)

    out_obj = {
        "experiment": exp_result,     # ExperimentResult (or None)
        "final_output": final_output, # SolverOutput for final phase
        "summary": summary,           # dict of scalar metrics (by level)
        "config": cfg.to_dict(),      # full effective config
    }

    config_path = args.config
    base_name = os.path.splitext(os.path.basename(config_path))[0]
    fname = os.path.join(save_dir, f"{base_name}_seed_{args.seed}.pkl")

    with open(fname, "wb") as f:
        pickle.dump(out_obj, f)

    print(f"Saved results to {fname}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config_from_args(args)

    # Override seed in cfg
    if args.seed is not None:
        if hasattr(cfg.pde, "data"):
            cfg.pde.data["seed"] = args.seed
        else:
            cfg.pde.seed = args.seed

    # Enable float64 if configured
    jax.config.update("jax_enable_x64", getattr(cfg.solver, "fp64", True))
    print("jax_enable_x64 =", jax.config.read("jax_enable_x64"))
    print(f"DATA TYPE SANITY CHECK: {jnp.ones((16,)).dtype}")
    print("Loaded config:")
    print(cfg)

    # Build PDE from config (only once!)
    p = build_pde_from_cfg(cfg)
    if not hasattr(p, "name"):
        p.name = getattr(cfg.pde, "name", "PDE")

    # Optional: set initial pad_size for kernel
    pad_size = getattr(cfg.pde, "init_pad_size", None)
    if pad_size is not None and hasattr(p, "kernel"):
        p.kernel.pad_size = pad_size

    # Run solver schedule (phases × homotopy levels with epsilon)
    all_outputs = run_solver_schedule(p, cfg)

    # Evaluate and summarize per level
    solver_global = SolverGlobalConfig.from_cfg(cfg)
    summary = {}

    for level_index, level_res in enumerate(all_outputs.levels):
        alpha = level_res.alpha
        epsilon = getattr(level_res, "epsilon", getattr(p, "epsilon", None))
        final_phase = level_res.phases[-1]
        final_alg_out = final_phase.output

        summary_level = evaluate_solution(
            p,
            solver_global,
            final_alg_out,
            level_index=level_index + 1,
            alpha=alpha,
            epsilon=epsilon,
            cfg=cfg,
            Ntest=getattr(cfg.pde, "Ntest", 100),
        )
        summary[level_index + 1] = summary_level

    final_alg_out = all_outputs.levels[-1].phases[-1].output

    save_results_pickle(
        exp_result=all_outputs,
        final_output=final_alg_out,
        summary=summary,
        cfg=cfg,
        args=args,
    )


if __name__ == "__main__":
    main()