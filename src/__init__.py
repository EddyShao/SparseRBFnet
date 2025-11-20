# src/__init__.py
"""
Public API for the src package.

We only expose subpackages (kernel, sovler, utils).
Users are expected to import concrete classes/functions from these
submodules, e.g.:

    from src.kernel.Kernels import GaussianKernel
    from src.sovler.solver import solve
"""

from . import kernel
from . import solver   # note: folder name is 'sovler'
from . import utils
from . import config

__all__ = ["kernel", "solver", "utils", "config"]