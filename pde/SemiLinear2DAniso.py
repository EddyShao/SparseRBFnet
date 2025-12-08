import numpy as np

from src.kernel.Kernels import GaussianKernel2DAnisotropic
from src.kernel.Kernels import KERNEL_BASE_REGISTRY
from src.utils import Objective, sample_cube_obs, plot_solution_2d

from typing import Dict, Callable, Any
import jax
import jax.numpy as jnp
from functools import partial

class SemiLinearKernelMixin:
    """
    Mixin with PDE-specific structure for the semilinear equation:
        E(u) = -\Delta u + u^3
    """

    def _init_semilinear_kernel(self, mask=False, D=None):
        """Call this in subclass __init__ after base-kernel init."""
        self.mask = mask
        self.D = D

        # linear results for computing E and B
        self.linear_E = (self.kappa_X_c_Xhat, self.Lap_kappa_X_c_Xhat)
        self.linear_B = (self.kappa_X_c_Xhat,)
        self.DE = (0,)
        self.DB = ()

    # ------------ kappa with optional mask ------------

    @partial(jax.jit, static_argnums=(0,))
    def kappa(self, x, s, xhat):
        out = super().kappa(x, s, xhat)
        if self.mask and self.D is not None:
            mask = jnp.prod(xhat - self.D[:, 0]) * jnp.prod(self.D[:, 1] - xhat)
            out = out * mask
        return out

    @partial(jax.jit, static_argnums=(0,))
    def kappa_X_c_Xhat(self, X, S, c, Xhat):
        return jax.vmap(self.kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    # ------------ Laplacian and PDE operator E, B ------------

    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c(self, X, S, c, xhat):
        return jnp.trace(jax.hessian(self.kappa_X_c, argnums=3)(X, S, c, xhat))

    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c_Xhat(self, X, S, c, Xhat):
        return jax.vmap(self.Lap_kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    @partial(jax.jit, static_argnums=(0,))
    def E_kappa_X_c(self, X, S, c, xhat):
        return -self.Lap_kappa_X_c(X, S, c, xhat) + self.kappa_X_c(X, S, c, xhat) ** 3

    @partial(jax.jit, static_argnums=(0,))
    def B_kappa_X_c(self, X, S, c, xhat):
        return self.kappa_X_c(X, S, c, xhat)

    # "vectorized" versions using precomputed linear results
    def E_kappa_X_c_Xhat(self, *linear_results):
        # linear_results = (kappa_X_c_Xhat, Lap_kappa_X_c_Xhat)
        return -linear_results[1] + linear_results[0] ** 3

    def B_kappa_X_c_Xhat(self, *linear_results):
        return linear_results[0]

    # ------------ derivatives wrt kernel parameters ------------

    @partial(jax.jit, static_argnums=(0,))
    def DE_kappa(self, x, s, xhat, *args):
        # args[0] could be kappa_X_c or similar; you used it in your original code
        return -jnp.trace(jax.hessian(self.kappa, argnums=2)(x, s, xhat)) \
               + 3 * args[0] ** 2 * self.kappa(x, s, xhat)

    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa(self, x, s, xhat, *args):
        return self.kappa(x, s, xhat)
    

class SemiLinearGaussianAniso2DKernel(SemiLinearKernelMixin, GaussianKernel2DAnisotropic):
    def __init__(self, d, power, sigma_max, sigma_min,
                 anisotropic=True, mask=False, D=None):
        GaussianKernel2DAnisotropic.__init__(
            self,
            d=d,
            power=power,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            anisotropic=anisotropic,
        )
        self._init_semilinear_kernel(mask=mask, D=D)



#########################################
#   Exact solution & RHS for this PDE   #
#########################################

def ex_sol_aniso(x):
    """
    Exact solution u(x) depending only on x[0]:

        u(x) = sin(2π x_0)
    """
    x = jnp.atleast_2d(x)  # (N, 2)
    result = jnp.sin(2.0 * jnp.pi * x[:, 0])
    return result if result.shape[0] > 1 else result[0]


def f_aniso(x):
    """
    Forcing f(x) such that

        -Δu + u^3 = f

    With u(x) = sin(2π x_0), we have
        -Δu = 4π^2 sin(2π x_0)
    so

        f(x) = 4π^2 sin(2π x_0) + u(x)^3
    """
    x = jnp.atleast_2d(x)
    result = 4.0 * jnp.pi**2 * jnp.sin(2.0 * jnp.pi * x[:, 0])
    result += ex_sol_aniso(x) ** 3
    return result if result.shape[0] > 1 else result[0]



#########################################
#            2D PDE class               #
#########################################

class PDE:
    """
    2D semilinear PDE on D = [-1,1]^2:

        -Δu + u^3 = f  in D
        u = g         on ∂D

    This is structured similarly to `SemiLinearHighDim.PDE`, but:
      - dimension fixed at d = 2
      - anisotropic ansatz with (theta, r1, r2) as extra parameters
      - NO kernel is built here; you will attach it yourself.
    """

    # registry for exact solutions / RHS choices (in case you add more later)

    KERNEL_REGISTRY: Dict[str, Callable[["PDE"], Any]] = {
        "gaussian": lambda kcfg, p: SemiLinearGaussianAniso2DKernel(
            d=p.d,
            D=p.D,
            power=p.power,
            sigma_max=kcfg.get('sigma_max', 1),
            sigma_min=kcfg.get('sigma_min', 1e-3),
            anisotropic=kcfg.get('anisotropic', True),
            mask=p.mask,
        ),
    }
    EXACT_SOL_REGISTRY = {
        "anisoSines": {
            "f": f_aniso,
            "ex_sol": ex_sol_aniso,
        },
    }

    def __init__(self, pcfg: dict, kcfg: dict):
        """
        pcfg : PDE config (from cfg.pde)
        kcfg : kernel config (from cfg.kernel) -- ignored here for now;
               you can use it later to build a kernel.
        """
        self.name = "SemiLinear2DAniso"

        # spatial dimension fixed at 2
        self.d = 2

        # power parameter (kept for consistency; you may use it in your kernel)
        self.power = float(pcfg.get("power", self.d + 2.01))

        # scaling & randomness
        self.scale = float(pcfg.get("scale", 1.0))
        self.mask = self.scale < 1e-8

        self.seed = int(pcfg.get("seed", 200))
        self.key = jax.random.PRNGKey(self.seed)

        # domain D = [-1,1]^2
        self.D = jnp.array([
            [-1.0, 1.0],
            [-1.0, 1.0],
        ])
        self.vol_D = jnp.prod(self.D[:, 1] - self.D[:, 0])

        # anisotropic ansatz: (x0, x1, theta, r1, r2)
        self.anisotropic = True
        self.dim = 5  # 2 spatial + 3 anisotropic parameters

        # parameter domain Ω for (x0, x1, theta, r1, r2)
        self.Omega = jnp.array([
            [-2.0,  2.0],   # x0
            [-2.0,  2.0],   # x1
            [-10.0, 10.0],  # theta
            [-7.0,  3.0],   # r1
            [-7.0,  3.0],   # r2
        ])

        # initial padded state (x, s, u)
        self.init_pad_size = int(pcfg.get("init_pad_size", 60))
        self.u_zero = {
            "x": jnp.zeros((self.init_pad_size, self.d)),      # centers
            "s": jnp.zeros((self.init_pad_size, 3)),           # [theta, r1, r2]
            "u": jnp.zeros((self.init_pad_size,)),             # outer weights
        }

        # ------------------------------------------------------------------
        # Observations (interior / boundary)
        # ------------------------------------------------------------------
        self.Nobs_int = pcfg.get("Nobs_int", None)
        self.Nobs_bnd = pcfg.get("Nobs_bnd", None)
        self.Nobs = pcfg.get("Nobs", None)  # fallback if above are None

        sampling_method = pcfg.get("sampling", "grid")

        if self.Nobs_int is not None and self.Nobs_bnd is not None:
            # new-style interface: explicit interior / boundary counts
            self.Nobs_int = int(self.Nobs_int)
            self.Nobs_bnd = int(self.Nobs_bnd)
            self.xhat_int, self.xhat_bnd = sample_cube_obs(
                self.D,
                self.Nobs_int,
                self.Nobs_bnd,
                method=sampling_method,
                rng=self.key,
            )
        else:
            # fallback: total Nobs, let sample_cube_obs decide the split
            Nobs = int(pcfg.get("Nobs", 50))
            self.xhat_int, self.xhat_bnd = sample_cube_obs(
                self.D,
                Nobs,
                method=sampling_method,
                rng=self.key,
            )
            self.Nobs_int = self.xhat_int.shape[0]
            self.Nobs_bnd = self.xhat_bnd.shape[0]

        self.xhat = jnp.vstack([self.xhat_int, self.xhat_bnd])
        self.Nx_int = self.xhat_int.shape[0]
        self.Nx_bnd = self.xhat_bnd.shape[0]
        self.Nx = self.Nx_int + self.Nx_bnd

        # For 2nd order PDE, only one boundary operator (Dirichlet),
        # so Objective takes (N_int, N_bnd)
        self.obj = Objective(self.Nx_int, self.Nx_bnd, scale=self.scale)

        # Test set: reuse training points by default
        self.test_int, self.test_bnd = self.xhat_int, self.xhat_bnd
        self.obj_test = Objective(
            self.test_int.shape[0], self.test_bnd.shape[0], scale=self.scale
        )

        # exact solution & RHS
        self.rhs_type = pcfg.get("rhs_type", "aniso_1d")
        self._build_exact_sol_rhs(self.rhs_type)

        # ------------------------------------------------------------------
        # Kernel placeholder
        # ------------------------------------------------------------------
        # you can later attach a kernel manually:
        #   self.kernel = YourKernel(...)
        self.kernel = None  # will be set by you

    # ------------------------------------------------------------------
    # Exact solution / RHS registry
    # ------------------------------------------------------------------
    def _build_exact_sol_rhs(self, ex_sol_key: str):
        reg = self.EXACT_SOL_REGISTRY
        if ex_sol_key not in reg:
            raise ValueError(
                f"Unknown exact_solution key '{ex_sol_key}'. "
                f"Available: {list(reg.keys())}"
            )

        self.f = reg[ex_sol_key]["f"]
        self.ex_sol = reg[ex_sol_key]["ex_sol"]


    # ------------------------------------------------------------------
    # Sampling routines
    # ------------------------------------------------------------------
    def sample_obs(self, Nobs_int, Nobs_bnd, method="grid"):
        """
        Exposed helper if you want to resample obs later.
        """
        obs_int, obs_bnd = sample_cube_obs(
            self.D,
            Nobs_int,
            Nobs_bnd,
            method=method,
            rng=self.key,
        )
        return obs_int, obs_bnd

    def sample_param(self, Ntarget):
        """
        Sample Ntarget random parameters in Ω.

        Returns:
            randomx : (Ntarget, 2)     center locations
            randoms : (Ntarget, 3)     [theta, r1, r2]
        """
        self.key, subkey1, subkey2, subkey3 = jax.random.split(self.key, 4)

        # x-centers
        randomx = self.Omega[: self.d, 0] + (
            self.Omega[: self.d, 1] - self.Omega[: self.d, 0]
        ) * jax.random.uniform(subkey1, shape=(Ntarget, self.d))

        # theta
        random_theta = self.Omega[self.d, 0] + (
            self.Omega[self.d, 1] - self.Omega[self.d, 0]
        ) * jax.random.uniform(subkey2, shape=(Ntarget, 1))

        # r1, r2 sampled with a shared u_r (as in your original code)
        u_r = jax.random.uniform(subkey3, shape=(Ntarget, 1))
        random_r1 = self.Omega[self.d + 1, 0] + (
            self.Omega[self.d + 1, 1] - self.Omega[self.d + 1, 0]
        ) * u_r
        random_r2 = self.Omega[self.d + 2, 0] + (
            self.Omega[self.d + 2, 1] - self.Omega[self.d + 2, 0]
        ) * u_r

        randoms = jnp.hstack([random_theta, random_r1, random_r2])  # (Ntarget, 3)
        return randomx, randoms

    # ------------------------------------------------------------------
    # Plotting stub (you can reuse your old plot_forward and plug kernel)
    # ------------------------------------------------------------------
    def plot_forward(self, x, s, c, suppc=None):
        """
        2D visualization (exact vs predicted, error contour).

        Left as a stub here since it depends on how you wire the kernel:

            Gu = self.kernel.gauss_X_c_Xhat(x, s, c, X_eval)

        You can copy your old plot_forward implementation and replace
        the kernel calls once you hook self.kernel.
        """
        pass