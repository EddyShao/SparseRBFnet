# src/pde/SemiLinearHighDim.py

import numpy as np

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Callable, Any

from src.kernel.Kernels import GaussianKernel, MaternKernel  # you can add Wendland/Matern later if desired
from src.utils import Objective, sample_cube_obs

# jax.config.update("jax_enable_x64", True)


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
        eps_diff: float = 0.1
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
        
        self.eps_diff = eps_diff


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
        return - (self.eps_diff**2) * lap_u + u_val**3 - u_val

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
        return - (self.eps_diff**2) * lap_u_vals + u_vals**3 - u_vals

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
        return - self.eps_diff**2 * u_val * temp + 3 * (args[0] ** 2) * u_val  - u_val

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
                 anisotropic=False, mask=False, D=None, nu=1.5, eps_diff=0.1):
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
            nu=nu,
        )
        self.mask = mask
        self.D = D

        # What your PDE expects for semilinear structure:
        self.linear_E = (self.kappa_X_c_Xhat, self.Lap_kappa_X_c_Xhat)
        self.linear_B = (self.kappa_X_c_Xhat,)
        self.DE = (0,)
        self.DB = ()
        self.eps_diff = eps_diff

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
        lap = self.Lap_kappa_X_c(X, S, c, xhat)
        kappa = self.kappa_X_c(X, S, c, xhat)
        return - self.eps_diff**2 * lap + kappa ** 3 - kappa

    @partial(jax.jit, static_argnums=(0,))
    def B_kappa_X_c(self, X, S, c, xhat):
        # boundary operator is just identity here
        return self.kappa_X_c(X, S, c, xhat)

    def E_kappa_X_c_Xhat(self, *linear_results):
        # linear_results = (kappa_X_c_Xhat, Lap_kappa_X_c_Xhat)
        return - self.eps_diff**2 * linear_results[1] + linear_results[0] ** 3 - linear_results[0]

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
        return - self.eps_diff**2 * lap_phi + 3.0 * (args[0] ** 2) * phi - phi

    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa(self, x, s, xhat, *args):
        return self.kappa(x, s, xhat)
    
def _ex_sol_tanh4d_single(x, eps=0.1):
    """
    Single-point version of the 4D tanh-layer solution:

        u(x) = tanh((x1 - 0.2)/eps) * tanh((x2 + 0.3)/eps)
               * sin(pi x3) * sin(pi x4)

    x: (d,) but we assume d >= 4 and only use the first four coordinates.
    """
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return (
        jnp.tanh((x1 - 0.2) / eps)
        * jnp.tanh((x2 + 0.3) / eps)
        * jnp.sin(jnp.pi * x3)
        * jnp.sin(jnp.pi * x4)
    )


def ex_sol_tanh4d(x, eps=0.1):
    r"""
    Exact 4D solution with internal layers:

        u(x) = tanh((x1 - 0.2)/eps) * tanh((x2 + 0.3)/eps)
               * sin(pi x3) * sin(pi x4),

    intended for d = 4 on D = [-1,1]^4.

    x: (..., d) or (d,)
    returns: scalar if input is (d,), or (N,) if input is (N, d)
    """
    x = jnp.atleast_2d(x)  # (N, d)

    vals = jax.vmap(lambda y: _ex_sol_tanh4d_single(y, eps))(x)  # (N,)
    return vals if vals.shape[0] > 1 else vals[0]


def _laplacian_tanh4d_single(x, eps=0.1):
    """
    Compute Δu(x) via trace of Hessian for the tanh4d solution.
    x: (d,) with d >= 4
    """
    def u_fun(z):
        return _ex_sol_tanh4d_single(z, eps)

    H = jax.hessian(u_fun)(x)  # (d, d)
    return jnp.trace(H)


def f_tanh4d(x, eps=0.1):
    r"""
    RHS for the semilinear 4D PDE

        -0.01 Δu(x) + u(x)^3 - u(x) = f(x),

    with
        u(x) = tanh((x1 - 0.2)/eps) * tanh((x2 + 0.3)/eps)
               * sin(pi x3) * sin(pi x4).

    x: (..., d) or (d,), intended for d = 4.
    returns: scalar if input is (d,), or (N,) if input is (N, d)
    """
    x = jnp.atleast_2d(x)  # (N, d)

    def f_single(y):
        u = _ex_sol_tanh4d_single(y, eps)
        lap_u = _laplacian_tanh4d_single(y, eps)  # Δu
        return -eps**2 * lap_u + u**3 - u

    vals = jax.vmap(f_single)(x)  # (N,)
    return vals if vals.shape[0] > 1 else vals[0]



def ex_sol_sine_easy(x):
    """
    u(x) = sum_i sin(pi * x_i)
    works for any dimension d
    """
    x = jnp.atleast_2d(x)
    return jnp.sum(jnp.sin(jnp.pi * x), axis=1) \
           if x.shape[0] > 1 else jnp.sum(jnp.sin(jnp.pi * x))


def f_sine_easy(x, eps=0.1):
    """
    f(x) = eps^2 * pi^2 * u(x) + u(x)^3 - u(x)
    matching the PDE:
        -eps^2 Δu + u^3 - u = f
    and Δu = -pi^2 u
    """
    u = ex_sol_sine_easy(x)
    return (eps**2) * (jnp.pi**2) * u + u**3 - u


def ex_sol_tanh_radial(x, eps=0.1, r0=0.25):
    r"""
    Radial Allen–Cahn-type layer:

        u(x) = tanh((||x|| - r0) / eps) + 1,

    where ||x|| = sqrt(sum_i x_i^2).
    Works for any spatial dimension d >= 1.
    """
    x = jnp.atleast_2d(x)            # (N, d)
    r = jnp.sqrt(jnp.sum(x**2, axis=1))  # (N,)
    xi = (r - r0) / eps
    u = jnp.tanh(xi) + 1.0
    return u if u.shape[0] > 1 else u[0]


def f_tanh_radial(x, eps=0.1, r0=0.25):
    r"""
    RHS for the semilinear PDE

        -eps^2 Δu(x) + u(x)^3 - u(x) = f(x),

    with
        u(x) = tanh((||x|| - r0) / eps) + 1.

    Works for any d >= 1.
    """
    x = jnp.atleast_2d(x)             # (N, d)
    d = x.shape[1]

    r = jnp.sqrt(jnp.sum(x**2, axis=1))   # (N,)
    r_safe = jnp.maximum(r, 1e-8)

    xi = (r - r0) / eps
    v = jnp.tanh(xi)               # unshifted tanh
    u = v + 1.0                   # full solution
    sech2 = 1.0 / jnp.cosh(xi)**2

    # φ'(r) = (1/eps) * sech^2(xi)
    phi_r = (1.0 / eps) * sech2
    # φ''(r) = -2/eps^2 * tanh(xi) * sech^2(xi)
    phi_rr = -2.0 / (eps**2) * v * sech2

    # Δu = φ''(r) + (d-1)/r * φ'(r)
    lap_u = phi_rr + (d - 1.0) / r_safe * phi_r

    # f = -eps^2 Δu + u^3 - u, with u = tanh(xi) + 1
    f = -(eps**2) * lap_u + u**3 - u
    return f if f.shape[0] > 1 else f[0]


def ex_sol_tanh_radial_2bump(
    x,
    eps=0.1,
    s1=10,
    r01=0.25,
    s2=1/10,
    r02=0.50,
    c1=None,
    c2=None,
):
    r"""
    Two-bump radial Allen–Cahn-type layer:

        u(x) = u1(x) + u2(x),
      where
        u1(x) = tanh((||x - c1|| - r01) / eps1) + 1,
        u2(x) = tanh((||x - c2|| - r02) / eps2) + 1.

    Default:
        - d is inferred from x (any d >= 1, typically d = 4),
        - c1 = (0.5, 0.5, ..., 0.5),
        - c2 = (-0.5, -0.5, ..., -0.5).

    x: (..., d) or (d,)
    returns: scalar if input is (d,), or (N,) if input is (N, d)
    """
    x = jnp.atleast_2d(x)  # (N, d)
    N, d = x.shape

    if c1 is None:
        c1 = jnp.ones(d) * 0.5
    if c2 is None:
        c2 = -jnp.ones(d) * 0.5

    # distances to each center
    r1 = jnp.sqrt(jnp.sum((x - c1) ** 2, axis=1))  # (N,)
    r2 = jnp.sqrt(jnp.sum((x - c2) ** 2, axis=1))  # (N,)

    xi1 = (r1 - r01) / (eps * s1)
    xi2 = (r2 - r02) / (eps * s2)

    v1 = jnp.tanh(xi1)
    v2 = jnp.tanh(xi2)

    u1 = v1 + 1.0
    u2 = v2 + 1.0

    u = u1 + u2  # (N,)
    return u if N > 1 else u[0]


def f_tanh_radial_2bump(
    x,
    eps=0.1,
    s1=10,
    r01=0.25,
    s2=1/10,
    r02=0.50,
    c1=None,
    c2=None,
):
    r"""
    RHS for the semilinear PDE

        -eps_diff^2 Δu(x) + u(x)^3 - u(x) = f(x),

    with two-bump exact solution

        u(x) = u1(x) + u2(x),
      where
        u1(x) = tanh((||x - c1|| - r01) / eps1) + 1,
        u2(x) = tanh((||x - c2|| - r02) / eps2) + 1.

    Works for any d >= 1 (typically d = 4).
    """
    x = jnp.atleast_2d(x)  # (N, d)
    N, d = x.shape

    if c1 is None:
        c1 = jnp.ones(d) * 0.5
    if c2 is None:
        c2 = -jnp.ones(d) * 0.5

    # distances to each center
    r1 = jnp.sqrt(jnp.sum((x - c1) ** 2, axis=1))  # (N,)
    r2 = jnp.sqrt(jnp.sum((x - c2) ** 2, axis=1))  # (N,)

    r1_safe = jnp.maximum(r1, 1e-8)
    r2_safe = jnp.maximum(r2, 1e-8)

    # local coordinates
    xi1 = (r1 - r01) / (eps * s1)
    xi2 = (r2 - r02) / (eps * s2)

    v1 = jnp.tanh(xi1)        # unshifted
    v2 = jnp.tanh(xi2)
    u1 = v1 + 1.0             # shifted
    u2 = v2 + 1.0

    u = u1 + u2               # total solution, shape (N,)

    sech2_1 = 1.0 / jnp.cosh(xi1) ** 2
    sech2_2 = 1.0 / jnp.cosh(xi2) ** 2

    # For each bump j:
    #   φ_j'(r_j)  = (1/eps_j) * sech^2(xi_j)
    #   φ_j''(r_j) = -2/eps_j^2 * tanh(xi_j) * sech^2(xi_j)

    phi1_r = (1.0 / (eps * s1)) * sech2_1
    phi2_r = (1.0 / (eps * s2)) * sech2_2

    phi1_rr = -2.0 / (eps * s1) ** 2 * v1 * sech2_1
    phi2_rr = -2.0 / (eps * s2) ** 2 * v2 * sech2_2

    # Δu1, Δu2 (radial formula per center)
    lap_u1 = phi1_rr + (d - 1.0) / r1_safe * phi1_r
    lap_u2 = phi2_rr + (d - 1.0) / r2_safe * phi2_r

    lap_u = lap_u1 + lap_u2  # total Laplacian

    # f = -eps_diff^2 Δu + u^3 - u
    f = -(eps ** 2) * lap_u + u ** 3 - u

    return f if N > 1 else f[0]


def _ex_sol_tanh_grid_single(x, eps=0.1, alpha=40.0):
    """
    Single-point version of a 'tanh grid' solution:

        u(x) = tanh(alpha * prod_i sin(pi x_i))

    eps is kept in the signature so that PDE._build_exact_sol_rhs
    can pass eps=self.eps_diff, but we don't actually use eps here.
    """
    g = jnp.prod(jnp.sin(jnp.pi * x))  # scalar
    return jnp.tanh(alpha * g)


def ex_sol_tanh_grid(x, eps=0.1, alpha=40.0):
    r"""
    Multi-layer internal structure:

        u(x) = tanh(alpha * prod_i sin(pi x_i)),

    which creates a grid of thin internal layers where sin(pi x_i) = 0.

    Works for any d >= 1. alpha controls the thickness of layers:
    larger alpha -> thinner layers.
    """
    x = jnp.atleast_2d(x)  # (N, d)

    vals = jax.vmap(lambda y: _ex_sol_tanh_grid_single(y, eps=eps, alpha=alpha))(x)
    return vals if vals.shape[0] > 1 else vals[0]


def _laplacian_tanh_grid_single(x, eps=0.1, alpha=40.0):
    """
    Compute Δu(x) = trace(Hessian(u)) for the tanh-grid solution at a single point.
    """

    def u_fun(z):
        return _ex_sol_tanh_grid_single(z, eps=eps, alpha=alpha)

    H = jax.hessian(u_fun)(x)  # (d, d)
    return jnp.trace(H)


def f_tanh_grid(x, eps=0.1, alpha=40.0):
    r"""
    RHS for

        -eps^2 Δu(x) + u(x)^3 - u(x) = f(x),

    with u(x) = tanh(alpha * prod_i sin(pi x_i)).

    eps is the diffusion scale (this should match p.eps_diff).
    """
    x = jnp.atleast_2d(x)  # (N, d)

    def f_single(y):
        u = _ex_sol_tanh_grid_single(y, eps=eps, alpha=alpha)
        lap_u = _laplacian_tanh_grid_single(y, eps=eps, alpha=alpha)
        return -eps**2 * lap_u + u**3 - u

    vals = jax.vmap(f_single)(x)
    return vals if vals.shape[0] > 1 else vals[0]




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
            eps_diff=p.eps_diff
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
            eps_diff=p.eps_diff
        ),
    }


    EXACT_SOL_REGISTRY = {
        'tanh4d': {
            'ex_sol': ex_sol_tanh4d,
            'f': f_tanh4d   
        },
        'sine_easy': {
            'ex_sol': ex_sol_sine_easy,
            'f': f_sine_easy
        },
        'tanh_radial': {
            'ex_sol': ex_sol_tanh_radial,
            'f': f_tanh_radial
        },
        'tanh_radial_2bump': {
            'ex_sol': ex_sol_tanh_radial_2bump,
            'f': f_tanh_radial_2bump
        },
        'tanh_grid': {
            'ex_sol': ex_sol_tanh_grid,
            'f': f_tanh_grid
        },
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

        self.eps_diff = float(pcfg.get("eps_diff", 0.1))

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
            entry = self.EXACT_SOL_REGISTRY[ex_sol_key]

            # keys where f/ex_sol expect an `eps` argument matching eps_diff
            if ex_sol_key in ("tanh4d", "tanh_radial", "tanh_radial_2bump", "tanh_grid"):
                # for tanh_grid, ex_sol ignores eps but having the same signature is convenient
                self.f = lambda x: entry["f"](x, eps=self.eps_diff)
                self.ex_sol = lambda x: entry["ex_sol"](x, eps=self.eps_diff)
            else:
                self.f = entry["f"]
                self.ex_sol = entry["ex_sol"]
        else:
            raise ValueError(
                f"Unknown exact_solution key '{ex_sol_key}'. "
                f"Available: {list(self.EXACT_SOL_REGISTRY.keys())}"
            )


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