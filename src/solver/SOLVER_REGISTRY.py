# src/solver_registry.py
from src.solver.solver import solve as solve_full
from src.solver.solver_outer import solve_outer
from src.solver.solver_outer_first_order import solve_outer_first_order

SOLVER_REGISTRY = {
    "full": solve_full,
    "outer": solve_outer,
    "outer_first_order": solve_outer_first_order,
}


