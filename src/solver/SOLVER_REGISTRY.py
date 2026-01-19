# src/solver_registry.py
from src.solver.solver import solve as solve_full
from src.solver.solver_outer import solve_outer
from src.solver.solver_outer_first_order import solve_outer_first_order
from src.solver.solver_aux import solve as solve_full_aux
from src.solver.solver_outer_aux import solve_outer as solve_outer_aux
from src.solver.solver_first import solve as solve_full_first
from src.solver.solver_no_boosting import solve as solve_no_boosting
from src.solver.solver_fixed_size import solve as solve_fixed_size
from src.solver.solver_fixed_size_smooth import solve as solve_fixed_size_smooth
from src.solver.solver_fixed_size_smooth_outer import solve as solve_fixed_size_smooth_outer
from src.solver.solver_fixed_size_outer import solve as solve_fixed_size_outer

SOLVER_REGISTRY = {
    "full": solve_full,
    "outer": solve_outer,
    "outer_aux": solve_outer_aux,
    "outer_first_order": solve_outer_first_order,
    "full_aux": solve_full_aux,
    "full_first": solve_full_first,
    "no_boosting": solve_no_boosting,
    "fixed_size": solve_fixed_size,
    "fixed_size_smooth": solve_fixed_size_smooth,
    "fixed_size_smooth_outer": solve_fixed_size_smooth_outer,
    "fixed_size_outer": solve_fixed_size_outer,
}


