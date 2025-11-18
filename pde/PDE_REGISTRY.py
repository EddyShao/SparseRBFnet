# src/pde_registry.py
from typing import Dict, Type
from src.config.base_config import Config

from pde.SemiLinearPDE import PDE as SemiLinearPDE
from pde.SemiLinearHighDim import PDE as SemiLinearHighDim
from pde.Eikonal import PDE as EikonalPDE
# from pde.OtherPDE import PDE as OtherPDE  # add if/when needed


PDE_CLASS_REGISTRY: Dict[str, Type] = {
    "SemiLinearPDE": SemiLinearPDE,
    "SemiLinearHighDim": SemiLinearHighDim,
}


def build_pde_from_cfg(cfg: Config):
    """
    Instantiate the PDE class specified by cfg.pde.cls, passing cfg.pde as a dict.
    DOES NOT set f or ex_sol – you can do that yourself or via a separate registry.
    """
    if not hasattr(cfg, "pde"):
        raise ValueError("Config has no 'pde' section.")

    pde_cfg: Config = cfg.pde
    kernel_cfg = getattr(cfg, "kernel", None)
    cls_name = getattr(pde_cfg, "cls", None)
    if cls_name is None:
        raise ValueError("cfg.pde.cls is not specified in the config.")

    if cls_name not in PDE_CLASS_REGISTRY:
        raise ValueError(
            f"Unknown PDE class '{cls_name}'. "
            f"Available: {list(PDE_CLASS_REGISTRY.keys())}"
        )

    PDECls = PDE_CLASS_REGISTRY[cls_name]
    p = PDECls(pde_cfg.to_dict(), kernel_cfg.to_dict())   # your existing PDE expects a dict
    return p