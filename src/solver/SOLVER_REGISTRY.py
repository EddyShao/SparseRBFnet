# src/solver_registry.py
from src.solver.solver import solve as solve_full
from src.solver.solver_outer import solve_outer
from src.solver.solver_outer_first_order import solve_outer_first_order
from src.solver.solver_aux import solve as solve_full_aux
from src.solver.solver_outer_aux import solve_outer as solve_outer_aux
from src.solver.solver_first import solve as solve_full_first

SOLVER_REGISTRY = {
    "full": solve_full,
    "outer": solve_outer,
    "outer_aux": solve_outer_aux,
    "outer_first_order": solve_outer_first_order,
    "full_aux": solve_full_aux,
    "full_first": solve_full_first,
}


