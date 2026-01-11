# src/pde/SemiLinearHighDim.py

import numpy as np

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Callable, Any

from src.kernel.Kernels import GaussianKernel, MaternKernel, WendlandKernel  # you can add Wendland/Matern later if desired
from src.utils import Objective, sample_cube_obs

# jax.config.update("jax_enable_x64", True)


###############################################################
# High-dimensional semilinear kernel (Gaussian only for now)  #
###############################################################

class RegressionGaussianHighDim(GaussianKernel):
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
        self.linear_E = (self.kappa_X_c_Xhat,)
        self.linear_B = (self.kappa_X_c_Xhat,)

        # flags for derivative structures used in your solver
        self.DE = ()
        self.DB = ()

    # ------------ PDE operators E, B ------------

    @partial(jax.jit, static_argnums=(0,))
    def E_kappa_X_c(self, X, S, c, xhat):
        return self.kappa_X_c(X, S, c, xhat)
        

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
        u_vals, = linear_results
        return u_vals

    def B_kappa_X_c_Xhat(self, *linear_results):
        u_vals, = linear_results
        return u_vals


    # ------------ derivatives wrt kernel parameters ------------

    @partial(jax.jit, static_argnums=(0,))
    def DE_kappa(self, x, s, xhat, *args):
        return self.kappa(x, s, xhat)

    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa(self, x, s, xhat, *args):
        """
        Derivative of B(u) wrt kernel parameters (identity here).
        """
        return self.kappa(x, s, xhat)

    

class RegressionMaternHighDim(MaternKernel):
    """
    High-dimensional Matérn kernel with hard-coded Laplacian and bi-Laplacian
    (via radial formulas + autodiff) for the semilinear bi-Laplacian PDE:
        E(u) = -Δ^2 u + u^3.
    """

    def __init__(
        self,
        d: int,
        nu: float,
        sigma_max: float,
        sigma_min: float,
        power: float = None,
        anisotropic: bool = False,
        mask: bool = False,
        D: jnp.ndarray = None,
        a: float = None,
    ):
        if power is None:
            # roughly "PDE-balanced" for 4th-order; feel free to tweak
            power = d + 4.01

        super().__init__(
            nu=nu,
            d=d,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            power=power,
        )
        self.anisotropic = anisotropic
        self.mask = mask
        self.D = D
        self.a = a # optional scaling coefficient for high order matern

        # linear results for computing E and B
        # here E uses Bi-Laplacian, B uses identity + Laplacian (aux)
        self.linear_E = (self.kappa_X_c_Xhat,)
        self.linear_B = (self.kappa_X_c_Xhat,)

        # flags for derivative structures used in your solver
        self.DE = ()
        self.DB = ()

    # -------- basic helpers: radial φ(r; ℓ), Lap φ, Δ² φ --------

    def _phi_radial(self, r, ell):
        
        """φ(r; ell) as a scalar radial function (used for autodiff)."""
        if self.a is not None:
            base = self.a ** self.d * self._matern_shape_iso(self.a * r, ell)
        else:
            base = self._matern_shape_iso(r, ell)
        factor = self._factor(ell)
        return factor * base

    def _phi_radial_first_second(self, r, ell):
        """Return φ'(r), φ''(r) via autodiff in the scalar r."""
        def f(rr):
            return self._phi_radial(rr, ell)

        f1 = jax.grad(f)(r)
        f2 = jax.grad(jax.grad(f))(r)
        return f1, f2

    def _radial_laplacian(self, r, ell):
        """
        Δφ(|x|) for a radial φ(|x|) in R^d:
            Δφ(r) = φ''(r) + (d-1)/r φ'(r),
        with a careful limit at r = 0.
        """
        f1, f2 = self._phi_radial_first_second(r, ell)
        eps = 1e-8

        # general formula
        lap = f2 + (self.d - 1.0) * f1 / (r + eps)

        # limit r -> 0: Δφ(0) = d * φ''(0)
        lap0 = self.d * f2
        return jnp.where(r < eps, lap0, lap)


    # -------- basic kernel evals over X / Xhat (if not already from _Kernel) --------

    @partial(jax.jit, static_argnums=(0,))
    def kappa_X(self, X, S, xhat):
        """
        φ(X_i, S_i; x̂) over centers X (N,d) for a single evaluation point xhat (d,).
        """
        return jax.vmap(self.kappa, in_axes=(0, 0, None))(X, S, xhat)

    @partial(jax.jit, static_argnums=(0,))
    def kappa_X_c(self, X, S, c, xhat):
        """
        Sum_i c_i φ(X_i, S_i; x̂).
        """
        phis = self.kappa_X(X, S, xhat)  # (Ncenters,)
        return jnp.dot(c, phis)

    @partial(jax.jit, static_argnums=(0,))
    def kappa_X_c_Xhat(self, X, S, c, Xhat):
        """
        Vectorized over x̂ ∈ Xhat, shape (Ntest, d).
        """
        return jax.vmap(self.kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)




    # ------------ PDE operators E, B ------------

    @partial(jax.jit, static_argnums=(0,))
    def E_kappa_X_c(self, X, S, c, xhat):

        return self.kappa_X_c(X, S, c, xhat)

    @partial(jax.jit, static_argnums=(0,))
    def B_kappa_X_c(self, X, S, c, xhat):
        return self.kappa_X_c(X, S, c, xhat)

    # "vectorized" versions using precomputed linear results
    def E_kappa_X_c_Xhat(self, *linear_results):
        # linear_results = (kappa_X_c_Xhat, BiLap_kappa_X_c_Xhat)
        u_vals, = linear_results
        return u_vals

    def B_kappa_X_c_Xhat(self, *linear_results):
        u_vals, = linear_results
        return u_vals

    # ------------ derivatives wrt kernel coefficients ------------

    @partial(jax.jit, static_argnums=(0,))
    def DE_kappa(self, x, s, xhat, *args):
        """
        Local derivative of E(u) wrt a single kernel coefficient (as in your Gaussian version):
            d/dc [ -Δ^2 u + u^3 ] = -Δ^2 κ + 3 u^2 κ
        where args[0] is the current u(x̂).
        """
        return self.kappa(x, s, xhat)

    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa(self, x, s, xhat, *args):
        """
        Derivative of B(u) wrt kernel coefficient (identity here).
        """
        return self.kappa(x, s, xhat)



def ex_sol_sum_low_freq(x):
    """
    Exact solution u(x) = sum_{i=1}^d sin((pi/4) * x_i)
    x: (..., d) or (d,)
    """
    x = jnp.atleast_2d(x)                          # (N, d)
    result = jnp.sum(jnp.sin(0.25 * jnp.pi * x), axis=1)
    return result if result.shape[0] > 1 else result[0]


def f_sum_low_freq(x):
    """
    RHS for Laplacian convention:
        Δu(x) = f(x)

    For u = sum_i sin((pi/4) x_i), we have Δu = -(pi/4)^2 u = -(pi^2/16) u.
    """
    return ex_sol_sum_low_freq(x)


def b_sum_low_freq(x):
    return ex_sol_sum_low_freq(x)



def ex_sol_sum(x):
    """
    Exact solution u(x) = sum_{i=1}^d sin((pi/2) * x_i)
    x: (..., d) or (d,)
    """
    x = jnp.atleast_2d(x)                          # (N, d)
    result = jnp.sum(jnp.sin(0.5 * jnp.pi * x), axis=1)
    return result if result.shape[0] > 1 else result[0]


def f_sum(x):
    """
    RHS for Laplacian convention:
        Δu(x) = f(x)

    For u = sum_i sin((pi/2) x_i), we have Δu = -(pi/2)^2 u = -(pi^2/4) u.
    """
    return ex_sol_sum(x)


def b_sum(x):
    return ex_sol_sum(x)




def _ex_sol_rpow4d_single(x, q=0.6):
    """
    x: (4,)
    returns: scalar u(x) = c * r(x)^q, where r(x) = ||x - x0||, x0=(0.5,...,0.5),
    and c = 2^{-q}.
    """
    r_squared = jnp.sum((x - 0.5) ** 2)
    c = 1 / (2 ** q)
    return c * (r_squared ** (q / 2.0))


def ex_sol_rpow4d(x, q=0.6):
    x = jnp.atleast_2d(x)  # (N, 4)
    vals = jax.vmap(lambda y: _ex_sol_rpow4d_single(y, q))(x)  # (N,)
    return vals if vals.shape[0] > 1 else vals[0]



def f_rpow4d(x, q=0.6):
    """
    RHS for Laplacian convention:
        Δu(x) = f(x)
    with u(x)=c*r(x)^q.
    """
    return ex_sol_rpow4d(x, q)


def b_rpow4d(x, q=0.6):
    return ex_sol_rpow4d(x, q)


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
        "gaussian": lambda kcfg, p: RegressionGaussianHighDim(
            d=p.d,
            power=p.power,
            sigma_max=kcfg.get("sigma_max", 1.0),
            sigma_min=kcfg.get("sigma_min", 1e-3),
            anisotropic=kcfg.get("anisotropic", False),
            mask=p.mask,
            D=p.D,
        ),
        'matern52': lambda kcfg, p: RegressionMaternHighDim(
            d=p.d,
            power=p.power,
            nu=kcfg.get("nu", 2.5),
            sigma_max=kcfg.get("sigma_max", 1.0),
            sigma_min=kcfg.get("sigma_min", 1e-3),
            anisotropic=kcfg.get("anisotropic", False),
            mask=p.mask,
            D=p.D,
            a=kcfg.get("a", None),
        ),
    }


    EXACT_SOL_REGISTRY = {
            "sum":{
                'f': f_sum,
                'ex_sol': ex_sol_sum,
            },
            "sum_low_freq":{
                'f': f_sum_low_freq,
                'ex_sol': ex_sol_sum_low_freq,
            },
            "rpow4d":{
                'f': f_rpow4d,
                'ex_sol': ex_sol_rpow4d,
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

        if kcfg.get("type", "gaussian") == "matern52" and kcfg.get("nu", 2.5) > 3:
            # for Matern ν=5/2 with anisotropic scaling,
            # use same scaling range for all dimensions
            self.Omega = self.Omega.at[self.d:, 0].set(-3.0)
            self.Omega = self.Omega.at[self.d:, 1].set(3.0)
        
        print(self.Omega)

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
        self._build_exact_sol_rhs(self.rhs_type, pcfg)

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

    def _build_exact_sol_rhs(self, ex_sol_key, pcfg=None):
        if ex_sol_key == 'rpow4d':
            if pcfg is not None:
                q = float(pcfg.get('q', 0.6))
            else:
                q = 0.6
            print(q)
            self.f = partial(f_rpow4d, q=q)
            self.ex_sol = partial(ex_sol_rpow4d, q=q)
        elif ex_sol_key in self.EXACT_SOL_REGISTRY:
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