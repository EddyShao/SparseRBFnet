import sys
sys.path.append("./")
import argparse
import os
import pickle

import jax.numpy as jnp
import numpy as np
from typing import Optional

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
from src.sovler.SOLVER_REGISTRY import SOLVER_REGISTRY
from src.utils import Objective, compute_errors, compute_rhs
from pde.PDE_REGISTRY import build_pde_from_cfg  



# ---------- Argument parsing ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Generic PDE solver driver.")
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
        help="Random seed, override config (if given).")
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override save_dir from YAML (optional).",
    )
    return parser.parse_args()




# ---------- Solver schedule (phases + homotopy) ----------

def build_alg_opts(global_cfg: SolverGlobalConfig,
                   phase_cfg: SolverPhaseConfig,
                   cfg) -> dict:
    """Build the flat dict of options expected by your solvers."""
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
    Run a single phase (e.g. warmup_outer or full refine).

    - Builds flat alg_opts from global + phase + PDE config.
    - Calls the registered solver.
    - Wraps the raw dict into SolverOutput.
    - Sets p.u_zero using the final *active* coefficients for warm start.
    """
    solver_fn = SOLVER_REGISTRY[phase_cfg.solver_name]
    alg_opts = build_alg_opts(global_cfg, phase_cfg, cfg)

    print(f"\n=== Phase: {phase_cfg.name} (solver={phase_cfg.solver_name}) ===")
    print(
        f"alpha={global_cfg.alpha:.3e}, T={global_cfg.T:.3e}, "
        f"max_step={phase_cfg.max_step}, TOL={phase_cfg.TOL:.1e}"
    )

    # Old raw dict from solver
    raw_alg_out = solver_fn(p, rhs, alg_opts)

    # Wrap into standardized SolverOutput
    alg_out = SolverOutput.from_raw(raw_alg_out)

    # Warm start: use *masked* final solution as p.u_zero for next phase
    x_last = alg_out.final["x"]
    s_last = alg_out.final["s"]
    c_last = alg_out.final["u"]

    # In case a solver returned no active support
    if x_last is None or s_last is None or c_last is None:
        # fall back to zeros
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
    Run [phase_1, phase_2, ...] once for a fixed (alpha, T),
    and wrap results into a LevelResult.

    Each phase becomes a PhaseResult containing:
      - name
      - solver_name
      - SolverOutput
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
    return level_res


def run_solver_schedule(p, rhs, cfg) -> ExperimentResult:
    """
    Top-level driver:
      - Build SolverGlobalConfig, HomotopyConfig, phase configs
      - Run phases x homotopy levels
      - Return an ExperimentResult object.
    """
    global_cfg = SolverGlobalConfig.from_cfg(cfg)
    hom_cfg = HomotopyConfig.from_cfg(cfg)

    phase_cfgs = [SolverPhaseConfig.from_cfg(ph) for ph in cfg.solver.phases]

    level_results: list[LevelResult] = []

    n_levels = hom_cfg.num_levels if hom_cfg.enabled else 1
    for level in range(n_levels):
        print(f"\n######## Homotopy level {level + 1}/{n_levels} ########")
        print(f"alpha = {global_cfg.alpha:.3e}, T = {global_cfg.T:.3e}")

        level_res = run_all_phases_one_level(
            p, rhs, global_cfg, phase_cfgs, cfg, level_index=level
        )
        level_results.append(level_res)

        # Update alpha/T for next level (if homotopy enabled)
        global_cfg.alpha *= hom_cfg.alpha_factor
        global_cfg.T *= hom_cfg.T_factor

    return ExperimentResult(levels=level_results)


# ---------- Evaluation & saving ----------

def evaluate_solution(
    p,
    rhs,
    solver_global: SolverGlobalConfig,
    alg_out: SolverOutput,
    level_index: int,
    alpha: float,
    *,
    Ntest=20
):
    """
    Compute train/test errors and residuals for the final solution.
    Assumes p.test_int, p.test_bnd, p.xhat_int, p.xhat_bnd, p.ex_sol, p.f exist.

    Uses the 'final' field of SolverOutput for evaluation.
    """
    # Build test RHS


    """
    Compute train/test errors and residuals for the final solution.

    If regenerate_test=True, we resample test points (and RHS) on the fly
    instead of using p.test_int / p.test_bnd created in __init__.
    """
    # -----------------------------
    # 1) Possibly regenerate test set
    # -----------------------------
    Ntest_int, Ntest_bnd = (Ntest-2)**p.d, 2 * p.d * (Ntest - 2) ** (p.d - 1)
    p.test_int, p.test_bnd = p.sample_obs(
        Ntest_int,
        Ntest_bnd,
        method='grid',
    )

    rhs_test_int = p.f(p.test_int)
    rhs_test_bnd = p.ex_sol(p.test_bnd)
    rhs_test = jnp.concatenate((rhs_test_int, rhs_test_bnd))

    # Objective for test set (uses same scale as training, or override)
    p.obj_test = Objective(
        p.test_int.shape[0],
        p.test_bnd.shape[0],
        scale=p.scale if hasattr(p, "scale") else 0.0,
    )

    # Final solution from SolverOutput
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

    # Train residual
    yk_train, _, _ = compute_rhs(
        p, xk_final, sk_final, ck_final, p.xhat_int, p.xhat_bnd
    )
    misfit_train = yk_train - rhs
    residue_train = p.obj.F(misfit_train)

    # Test residual
    yk_test, _, _ = compute_rhs(
        p, xk_final, sk_final, ck_final, p.test_int, p.test_bnd
    )
    misfit_test = yk_test - rhs_test
    residue_test = p.obj_test.F(misfit_test)

    print()
    print("#" * 20)
    print(f"alpha (level {level_index}) ~ {alpha:.1e}")
    print(
        "L_inf error test (boundary): {L_inf_bnd_test:.2e}\n"
        "L_inf error test (interior): {L_inf_int_test:.2e}\n"
        "L_inf error test (total): {L_inf_test:.2e}\n"
        "L_2 error test: {L_2_test:.2e}".format(**errors_test)
    )
    print(f"residue test: {residue_test:.2e}")
    print(
        "L_inf error train (boundary): {L_inf_bnd_train:.2e}\n"
        "L_inf error train (interior): {L_inf_int_train:.2e}\n"
        "L_inf error train (total): {L_inf_train:.2e}\n"
        "L_2 error train: {L_2_train:.2e}".format(**errors_train)
    )
    print(f"residue train: {residue_train:.2e}")
    print(f"final support: {alg_out.meta['final_supp']}")
    print("#" * 20)
    print()

    summary = {
        "alpha": alpha,
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

    Parameters
    ----------
    p : PDE object
        Contains PDE metadata such as `name`.
    exp_result : ExperimentResult or None
        Full homotopy/phase hierarchy (may be None if you only ran a single phase).
    final_output : SolverOutput
        Output of the final phase in the final homotopy level.
    summary : dict
        Numeric summary of the final solution (errors, residuals, etc.).
    cfg : Config
        Full configuration object (with `.to_dict()` method).
    args : argparse.Namespace
        Command-line arguments (for save_dir override).
    """


    save_dir = args.save_dir 

    if save_dir is None:
        print("No save_dir specified (neither CLI nor config.io.save_dir); skipping saving.")
        return

    # Optional: save_idx from config.io (if present)


    os.makedirs(save_dir, exist_ok=True)

    # ----- Object to be pickled -----
    out_obj = {
        "experiment": exp_result,          # ExperimentResult (or None)
        "final_output": final_output,      # SolverOutput for final phase
        "summary": summary,          # dict of scalar metrics
        "config": cfg.to_dict(),           # full effective config
    }
    # extract the config filename as the basename, the things between last '/' and '.yaml' or '.yml'
    config_path = args.config
    base_name = os.path.splitext(os.path.basename(config_path))[0]

    fname = os.path.join(
        save_dir,
        f"{base_name}_seed_{args.seed}.pkl",
    )
    with open(fname, "wb") as f:
        pickle.dump(out_obj, f)

    print(f"Saved results to {fname}")


# ---------- Main ----------

def main():
    args = parse_args()
    cfg = load_config_from_args(args)
    if args.seed is not None:
        # assuming Config stores underlying dict in .data or similar
        cfg.pde.data["seed"] = args.seed
        
    print("Loaded config:")
    print(cfg)


    # Build PDE from config
    p = build_pde_from_cfg(cfg)

    # If your pde_registry doesn't set these, you can still override here:
    # p.f = ...
    # p.ex_sol = ...
    # p.name = "SomeName"
    if not hasattr(p, "name"):
        p.name = getattr(cfg.pde, "name", "PDE")

    # Optional: pad_size from config
    pad_size = getattr(cfg.pde, "init_pad_size", None)
    if pad_size is not None and hasattr(p, "kernel"):
        p.kernel.pad_size = pad_size

    # Build RHS (train)
    rhs = p.f(p.xhat)
    if getattr(cfg.pde, "add_noise", False):
        rhs_mag = jnp.max(jnp.abs(rhs[:-p.Nx_bnd]))
        noise = jnp.random.randn(p.Nx) * 0.01 * rhs_mag
        rhs = rhs + noise
    rhs = rhs.at[-p.Nx_bnd:].set(p.ex_sol(p.xhat_bnd))

    # Run solver schedule (phases × homotopy levels)
    all_outputs = run_solver_schedule(p, rhs, cfg)


    # Evaluate and save
    solver_global = SolverGlobalConfig.from_cfg(cfg)

    summary = {}
    print(len(all_outputs.levels))
    for level_index, level_res in enumerate(all_outputs.levels):
        alpha = level_res.alpha
        final_phase = level_res.phases[-1]
        final_alg_out = final_phase.output
        summary_level = evaluate_solution(p, rhs, solver_global, final_alg_out, level_index=level_index+1, alpha=alpha, 
                                        Ntest=getattr(cfg.pde, 'Ntest', 100))
        summary[level_index+1] = summary_level
    
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