from dataclasses import dataclass
from typing import Any, Dict, List
from .base_config import Config
import numpy as np


@dataclass
class SolverPhaseConfig:
    name: str
    solver_name: str
    max_step: int
    TOL: float
    plot_final: bool

    @classmethod
    def from_cfg(cls, phase_cfg: Config) -> "SolverPhaseConfig":
        return cls(
            name=str(phase_cfg.name),
            solver_name=str(phase_cfg.solver_name),
            max_step=int(phase_cfg.max_step),
            TOL=float(phase_cfg.TOL),
            plot_final=bool(getattr(phase_cfg, "plot_final", True)),
        )


@dataclass
class SolverGlobalConfig:
    alpha: float
    T: float
    gamma: float
    Ntrial: int
    blocksize: int
    print_every: int
    plot_every: int

    @classmethod
    def from_cfg(cls, cfg: Config) -> "SolverGlobalConfig":
        s = cfg.solver
        return cls(
            alpha=float(s.alpha),
            T=float(s.T),
            gamma=float(s.gamma),
            Ntrial=int(s.Ntrial),
            blocksize=int(s.blocksize),
            print_every=int(s.print_every),
            plot_every=int(s.plot_every),
        )


@dataclass
class HomotopyConfig:
    enabled: bool
    num_levels: int
    alpha_factor: float
    T_factor: float

    @classmethod
    def from_cfg(cls, cfg: Config) -> "HomotopyConfig":
        h = getattr(cfg.solver, "homotopy", Config({}))
        return cls(
            enabled=bool(getattr(h, "enabled", False)),
            num_levels=int(getattr(h, "num_levels", 1)),
            alpha_factor=float(getattr(h, "alpha_factor", 1.0)),
            T_factor=float(getattr(h, "T_factor", 1.0)),
        )
    

