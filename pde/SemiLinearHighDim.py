# src/pde/SemiLinearHighDim.py

import numpy as np

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Callable, Any

from src.kernel.Kernels import GaussianKernel, MaternKernel  # you can add Wendland/Matern later if desired
from src.utils import Objective, sample_cube_obs

jax.config.update("jax_enable_x64", True)


###############################################################
# High-dimensional semilinear kernel (Gaussian only for now)  #
###############################################################

class SemiLinearHighDimGaussianKernel(GaussianKernel):
    """
    High-dimensional Gaussian kernel with hard-coded Laplacian
    for the semilinear PDE:
        E(u) = -Δu + u^3.

    This is essentially your old `Kernel` class, just renamed
    and slightly cleaned up.
    """

    def __init__(
        self,
        d: int,
        power: float,
        sigma_max: float,
        sigma_min: float,
        anisotropic: bool = False,
        mask: bool = False,
        D: jnp.ndarray = None,
    ):
        super().__init__(
            d=d,
            power=power,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            anisotropic=anisotropic,
        )
        self.mask = mask
        self.D = D

        # linear results for computing E and B
        self.linear_E = (self.kappa_X_c_Xhat, self.Lap_kappa_X_c_Xhat)
        self.linear_B = (self.kappa_X_c_Xhat,)

        # flags for derivative structures used in your solver
        self.DE = (0,)
        self.DB = ()


    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c(self, X, S, c, xhat):
        """
        Hard-coded Laplacian in x̂ for the Gaussian kernel:

            Δ_{x̂} (sum_i c_i κ(X_i, S_i; x̂))

        This is your old code, generalized to JAX.
        """
        diff = X - xhat          # shape (Ncenters, d)
        squared_diff = jnp.sum(diff ** 2, axis=1)  # shape (Ncenters,)

        sigma = self.sigma(S).squeeze()  # shape (Ncenters,)
        # temp = (|x - x̂|^2 - d σ^2) / σ^4
        temp = (squared_diff - self.d * sigma**2) / (sigma**4)

        # φ_i(x̂) = κ(X_i, S_i; x̂)
        phis = self.kappa_X(X, S, xhat)  # shape (Ncenters,)
        lap_phis = phis * temp           # shape (Ncenters,)

        return jnp.dot(c, lap_phis)      # scalar

    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c_Xhat(self, X, S, c, Xhat):
        """
        Vectorized Laplacian over x̂.
        """
        return jax.vmap(self.Lap_kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    # ------------ PDE operators E, B ------------

    @partial(jax.jit, static_argnums=(0,))
    def E_kappa_X_c(self, X, S, c, xhat):
        """
        E(u) = -Δu + u^3 applied to kernel expansion.
        """
        u_val = self.kappa_X_c(X, S, c, xhat)
        lap_u = self.Lap_kappa_X_c(X, S, c, xhat)
        return -lap_u + u_val**3

    @partial(jax.jit, static_argnums=(0,))
    def B_kappa_X_c(self, X, S, c, xhat):
        """
        Boundary operator B(u).
        For now: identity (Dirichlet).
        """
        return self.kappa_X_c(X, S, c, xhat)

    # "vectorized" versions using precomputed linear results
    def E_kappa_X_c_Xhat(self, *linear_results):
        # linear_results = (kappa_X_c_Xhat, Lap_kappa_X_c_Xhat)
        u_vals, lap_u_vals = linear_results
        return -lap_u_vals + u_vals**3

    def B_kappa_X_c_Xhat(self, *linear_results):
        (u_vals,) = linear_results
        return u_vals

    # ------------ derivatives wrt kernel parameters ------------

    @partial(jax.jit, static_argnums=(0,))
    def DE_kappa(self, x, s, xhat, *args):
        """
        Derivative of E(u) wrt kernel parameters as in your old code.
        """
        diff = x - xhat
        squared_diff = jnp.sum(diff ** 2)
        sigma = self.sigma(s).squeeze()

        temp = (squared_diff - self.d * sigma**2) / (sigma**4)
        u_val = self.kappa(x, s, xhat)  # scalar
        # args[0] is typically κ_X_c or similar, following your old interface
        return -u_val * temp + 3 * (args[0] ** 2) * u_val

    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa(self, x, s, xhat, *args):
        """
        Derivative of B(u) wrt kernel parameters (identity here).
        """
        return self.kappa(x, s, xhat)
    



class SemiLinearHighDimMaternKernel(MaternKernel):
    """
    High-dimensional Matérn(ν=3/2) kernel with analytic Laplacian for
    the semilinear PDE E(u) = -Δu + u^3.

    Assumes:
      - Base MaternKernel implements:
          * self.kappa(x, s, xhat)        # scalar
          * self.kappa_X(X, S, xhat)      # (N_centers,)
          * self.sigma(S)                 # length scale(s)
      - self.d is the spatial dimension.
    """

    def __init__(self, d, power, sigma_max, sigma_min,
                 anisotropic=False, mask=False, D=None, nu=1.5):
        """
        Parameters
        ----------
        d : int
            Spatial dimension.
        power : float
            Your usual 'power' parameter (you already pass this to MaternKernel).
        sigma_max, sigma_min : float
            Scale range (for self.sigma).
        anisotropic : bool
            If you want anisotropic sigmas; handled by base MaternKernel.
        mask : bool
            Whether to enforce zero outside D via a simple box mask.
        D : array (d, 2) or None
            Domain box [lo_i, hi_i] in each dimension.
        nu : float
            Matérn smoothness; here we assume ν=1.5.
        """
        super().__init__(
            d=d,
            power=power,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            anisotropic=anisotropic,
            nu=nu,
        )
        self.mask = mask
        self.D = D

        # What your PDE expects for semilinear structure:
        self.linear_E = (self.kappa_X_c_Xhat, self.Lap_kappa_X_c_Xhat)
        self.linear_B = (self.kappa_X_c_Xhat,)
        self.DE = (0,)
        self.DB = ()

    # ---------------- core kappa with optional mask ----------------

    @partial(jax.jit, static_argnums=(0,))
    def kappa(self, x, s, xhat):
        out = super().kappa(x, s, xhat)
        if self.mask and self.D is not None:
            mask = jnp.prod(xhat - self.D[:, 0]) * jnp.prod(self.D[:, 1] - xhat)
            out = out * mask
        return out


    # ---------------- Laplacian: analytic Matérn(ν=3/2) ----------------

    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c(self, X, S, c, xhat):
        """
        Compute Δ_x [ sum_i c_i k(||x - x_i||, σ_i) ] at x = xhat,
        using the analytic formula for Matérn(ν=3/2).
        """
        # diff: (N_centers, d)
        diff = X - xhat  # broadcasted
        r = jnp.sqrt(jnp.sum(diff ** 2, axis=1)) + jnp.finfo(jnp.float32).eps  # (N_centers,)

        # length scale(s) from S; same shape as number of centers
        ell = self.sigma(S).squeeze()   # (N_centers,)

        # Matérn(3/2): k(r) = (1 + a r) exp(-a r), with a = sqrt(3)/ell
        sqrt3 = jnp.sqrt(3.0)
        a = sqrt3 / ell                  # (N_centers,)

        # φ_i(xhat) from base class (already k(r))
        phi = self.kappa_X(X, S, xhat)   # (N_centers,)

        # Laplacian of φ: Δφ = a^2 * e^{-a r} * (a r - d)
        # and e^{-a r} = φ / (1 + a r)
        # => Δφ = a^2 * φ * (a r - d) / (1 + a r)
        lap_phis = a**2 * phi * (a * r - self.d) / (1.0 + a * r)

        # sum_i c_i Δφ_i
        return jnp.dot(c, lap_phis)

    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c_Xhat(self, X, S, c, Xhat):
        # vectorized over Xhat
        return jax.vmap(self.Lap_kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    # ---------------- PDE operators E, B ----------------

    @partial(jax.jit, static_argnums=(0,))
    def E_kappa_X_c(self, X, S, c, xhat):
        # E(u) = -Δu + u^3
        return - self.Lap_kappa_X_c(X, S, c, xhat) + self.kappa_X_c(X, S, c, xhat) ** 3

    @partial(jax.jit, static_argnums=(0,))
    def B_kappa_X_c(self, X, S, c, xhat):
        # boundary operator is just identity here
        return self.kappa_X_c(X, S, c, xhat)

    def E_kappa_X_c_Xhat(self, *linear_results):
        # linear_results = (kappa_X_c_Xhat, Lap_kappa_X_c_Xhat)
        return - linear_results[1] + linear_results[0] ** 3

    def B_kappa_X_c_Xhat(self, *linear_results):
        return linear_results[0]

    # ---------------- derivative wrt kernel parameters ----------------

    @partial(jax.jit, static_argnums=(0,))
    def DE_kappa(self, x, s, xhat, *args):
        """
        Derivative of E with respect to the kernel parameters (for Gauss-Newton).

        Uses analytic Δ κ(x, s, xhat), same Matérn(3/2) formula.
        args[0] is typically u(xhat) = kappa_X_c(X,S,c,xhat) used in your code.
        """
        # single-center version of the Laplacian formula
        diff = x - xhat
        r = jnp.sqrt(jnp.sum(diff ** 2)) + 1e-12

        ell = self.sigma(s).squeeze()
        sqrt3 = jnp.sqrt(3.0)
        a = sqrt3 / ell

        phi = self.kappa(x, s, xhat)  # scalar
        lap_phi = a**2 * phi * (a * r - self.d) / (1.0 + a * r)

        # E φ = -Δφ + (u(x))^3 φ, where u(x) ≈ args[0]
        return -lap_phi + 3.0 * (args[0] ** 2) * phi

    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa(self, x, s, xhat, *args):
        return self.kappa(x, s, xhat)
    




def ex_sol_prod(x):
    x = jnp.atleast_2d(x)
    result = jnp.prod(jnp.sin(jnp.pi * x), axis=1) 
    return result if len(result) > 1 else result[0]

def f_prod(x):
    d = x.shape[-1]
    return d * jnp.pi**2 * ex_sol_prod(x) + ex_sol_prod(x) ** 3

def ex_sol_sum (x):
    """
    Exact solution u(x) = sum_{i=1}^d sin(pi/2 * x_i)
    x: (..., d) or (d,)
    """
    x = jnp.atleast_2d(x)                          # (N, d)
    result = jnp.sum(jnp.sin(0.5 * jnp.pi * x), axis=1)
    return result if result.shape[0] > 1 else result[0]


def f_sum(x):
    """
    RHS f(x) = -Δu(x) = (pi^2 / 4) * sum_{i=1}^d sin(pi/2 * x_i)
             = (pi^2 / 4) * u(x)
    """
    return (jnp.pi**2 / 4.0) * ex_sol_sum(x) + ex_sol_sum(x) ** 3


#########################################
#       High-dimensional PDE class      #
#########################################

class PDE:
    """
    High-dimensional semilinear PDE:

        -\Delta u + u^3 = f   in D = [-1,1]^d
        u = g           on \partial D

    This version is structured like SemiLinearPDE but for general d,
    and uses the Gaussian kernel with analytic Laplacian.
    """

    # For now, only Gaussian is supported here because we need
    # kernel-specific Laplacians. You can add Wendland/Matern later.
    KERNEL_REGISTRY: Dict[str, Callable[[dict, "PDE"], Any]] = {
        "gaussian": lambda kcfg, p: SemiLinearHighDimGaussianKernel(
            d=p.d,
            power=p.power,
            sigma_max=kcfg.get("sigma_max", 1.0),
            sigma_min=kcfg.get("sigma_min", 1e-3),
            anisotropic=kcfg.get("anisotropic", False),
            mask=p.mask,
            D=p.D,
        ),
        'matern32': lambda kcfg, p: SemiLinearHighDimMaternKernel(
            d=p.d,
            power=p.power,
            nu=kcfg.get("nu", 1.5),
            sigma_max=kcfg.get("sigma_max", 1.0),
            sigma_min=kcfg.get("sigma_min", 1e-3),
            anisotropic=kcfg.get("anisotropic", False),
            mask=p.mask,
            D=p.D,
        ),
    }


    EXACT_SOL_REGISTRY = {
            "sum":{
                'f': f_sum,
                'ex_sol': ex_sol_sum
            },
            
            "prod": {
                'f': f_prod,
                'ex_sol': ex_sol_prod
            }
        }

    def __init__(self, pcfg: dict, kcfg: dict):
        """
        pcfg : config for PDE (from cfg.pde)
        kcfg : config for kernel (from cfg.kernel)
        """
        self.name = "SemiLinearHighDim"

        # dimension, power and scaling
        self.d = int(pcfg.get("d", 4))
        self.power = float(pcfg.get("power", self.d + 2.01))
        self.scale = float(pcfg.get("scale", 1.0))
        self.mask = self.scale < 1e-8

        self.seed = int(pcfg.get("seed", 200))
        self.key = jax.random.PRNGKey(self.seed)

        # domain D = [-1,1]^d
        self.D = jnp.zeros((self.d, 2))
        self.D = self.D.at[:, 0].set(-1.0)
        self.D = self.D.at[:, 1].set(1.0)
        self.vol_D = jnp.prod(self.D[:, 1] - self.D[:, 0])

        # kernel
        self.kernel = self._build_kernel(kcfg)
        self.anisotropic = bool(kcfg.get("anisotropic", False))

        # parameter dimension
        if self.anisotropic:
            self.dim = 2 * self.d
        else:
            self.dim = self.d + 1

        # parameter domain Ω
        self.Omega = jnp.zeros((self.dim, 2))
        self.Omega = self.Omega.at[:self.d, 0].set(-2.0)
        self.Omega = self.Omega.at[:self.d, 1].set(2.0)
        self.Omega = self.Omega.at[self.d:, 0].set(-10.0)
        self.Omega = self.Omega.at[self.d:, 1].set(0.0)

        if self.anisotropic:
            self.Omega = jnp.vstack(
                [self.Omega[:self.d, :], jnp.tile(self.Omega[self.d, :], (self.d, 1))]
            )

        assert self.dim == self.Omega.shape[0] and self.d == self.D.shape[0]

        # initial guess (padded)
        self.init_pad_size = int(pcfg.get("init_pad_size", 16))
        self.u_zero = {
            "x": jnp.zeros((self.init_pad_size, self.d)),
            "s": jnp.zeros((self.init_pad_size, self.dim - self.d)),
            "u": jnp.zeros((self.init_pad_size,)),
        }

        # observations
        self.Nobs_int = pcfg.get("Nobs_int", None)
        self.Nobs_bnd = pcfg.get("Nobs_bnd", None)

        if self.Nobs_int is None or self.Nobs_bnd is None:
            # fall back to Nobs if present, otherwise error
            Nobs = pcfg.get("Nobs", None)
            if Nobs is None:
                raise ValueError(
                    "SemiLinearHighDim: please provide either (Nobs_int, Nobs_bnd) "
                    "or a total Nobs in the config."
                )
            Nobs = int(Nobs)
            self.Nobs_int = int((Nobs - 2) ** self.d)
            self.Nobs_bnd = 2 * self.d * (Nobs - 2) ** (self.d - 1)

        self.xhat_int, self.xhat_bnd = self.sample_obs(
            self.Nobs_int,
            self.Nobs_bnd,
            method=pcfg.get("sampling", "grid"),
        )
        self.xhat = jnp.vstack([self.xhat_int, self.xhat_bnd])
        self.Nx_int = self.xhat_int.shape[0]
        self.Nx_bnd = self.xhat_bnd.shape[0]
        self.Nx = self.Nx_int + self.Nx_bnd

        self.obj = Objective(self.Nx_int, self.Nx_bnd, scale=self.scale)

        # For now, test = train (you can change later)
        self.test_int, self.test_bnd = self.xhat_int, self.xhat_bnd
        self.obj_test = Objective(
            self.test_int.shape[0], self.test_bnd.shape[0], scale=self.scale
        )

        # exact solution and rhs
        self.rhs_type = pcfg.get('rhs_type', 'sines')
        self._build_exact_sol_rhs(self.rhs_type)

    # ---------- helpers ----------

    def _build_kernel(self, kcfg: dict):
        ktype = kcfg.get("type", "gaussian")
        if ktype not in self.KERNEL_REGISTRY:
            raise ValueError(
                f"Unknown kernel_type '{ktype}' for SemiLinearHighDim. "
                f"Available: {list(self.KERNEL_REGISTRY.keys())}"
            )
        builder = self.KERNEL_REGISTRY[ktype]
        return builder(kcfg, self)

    def _build_exact_sol_rhs(self, ex_sol_key):
        if ex_sol_key in self.EXACT_SOL_REGISTRY:
            self.f = self.EXACT_SOL_REGISTRY[ex_sol_key]['f']
            self.ex_sol = self.EXACT_SOL_REGISTRY[ex_sol_key]['ex_sol']
        else:
            raise ValueError(f"Unknown exact_solution key '{ex_sol_key}'. Available: {list(self.EXACT_SOL_REGISTRY.keys())}")

    # ---------- PDE interface ----------

    def sample_obs(self, Nobs_int, Nobs_bnd, method="grid"):
        """
        Sample interior and boundary points in D.
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
        Sample random parameters in Ω.
        """
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)

        randomx = self.Omega[: self.d, 0] + (
            self.Omega[: self.d, 1] - self.Omega[: self.d, 0]
        ) * jax.random.uniform(subkey1, shape=(Ntarget, self.d))

        randoms = self.Omega[-1, 0] + (
            self.Omega[self.d :, 1] - self.Omega[self.d :, 0]
        ) * jax.random.uniform(subkey2, shape=(Ntarget, self.dim - self.d))

        return randomx, randoms

    def plot_forward(self, x, s, c, suppc=None):
        """
        High-dimensional visualization is not considered at the moment.
        """
        pass