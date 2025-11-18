from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np


# Saving configs

@dataclass
class SolverOutput:
    """
    Standardized container for a single solver run (one phase).
    """
    history: Dict[str, Any]
    final: Dict[str, Any]
    meta: Dict[str, Any]

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "SolverOutput":
        """
        Build a SolverOutput from the 'old style' alg_out dict
        returned by the solver.
        """
        history_keys = ["xk", "sk", "ck", "suppc", "js", "tics"]
        history = {k: raw[k] for k in history_keys if k in raw}

        xk_seq = history.get("xk", [])
        sk_seq = history.get("sk", [])
        ck_seq = history.get("ck", [])
        supp_seq = history.get("suppc", [])

        if xk_seq and sk_seq and ck_seq and supp_seq:
            supp_final = np.array(supp_seq[-1])
            x_final_full = np.array(xk_seq[-1])
            s_final_full = np.array(sk_seq[-1])
            c_final_full = np.array(ck_seq[-1])

            final_x = x_final_full[supp_final]
            final_s = s_final_full[supp_final]
            final_u = c_final_full[supp_final]

            final = {
                "x": final_x,
                "s": final_s,
                "u": final_u,
                "support": supp_final,
            }
            num_iter = len(xk_seq)
            final_supp_size = int(supp_final.sum())
        else:
            final = {
                "x": None,
                "s": None,
                "u": None,
                "support": None,
            }
            num_iter = 0
            final_supp_size = 0

        if "tics" in history and len(history["tics"]) > 0:
            runtime = float(history["tics"][-1])
        else:
            runtime = 0.0

        meta = {
            "num_iter": num_iter,
            "runtime": runtime,
            "final_supp": final_supp_size,
        }

        return cls(history=history, final=final, meta=meta)


@dataclass
class PhaseResult:
    """
    Result of one phase (e.g. warmup_outer or refine_full)
    at a given homotopy level.
    """
    name: str
    solver_name: str
    output: SolverOutput


@dataclass
class LevelResult:
    """
    Result of one homotopy level: fixed (alpha, T),
    with multiple phases executed in sequence.
    """
    level_index: int       # 0-based
    alpha: float
    T: float
    phases: List[PhaseResult]


@dataclass
class ExperimentResult:
    """
    Full experiment: multiple homotopy levels, each with several phases.
    """
    levels: List[LevelResult]