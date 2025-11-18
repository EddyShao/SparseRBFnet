# import jax.numpy as np
import numpy as np

from src.kernel.Kernels import GaussianKernel, WendlandKernel, MaternKernel
from src.kernel.Kernels import KERNEL_BASE_REGISTRY
from src.utils import Objective, sample_cube_obs, plot_solution_2d

from typing import Dict, Callable, Any
import jax
import jax.numpy as jnp
from functools import partial
jax.config.update("jax_enable_x64", True)

###############################################################
############ SemiLinear Kernel Mixin and Kernels ##############
################################################################
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
    

# ------------ SemiLinear Kernels ------------    
class SemiLinearGaussianKernel(SemiLinearKernelMixin, GaussianKernel):
    def __init__(self, d, power, sigma_max, sigma_min,
                 anisotropic=False, mask=False, D=None):
        GaussianKernel.__init__(
            self,
            d=d,
            power=power,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            anisotropic=anisotropic,
        )
        self._init_semilinear_kernel(mask=mask, D=D)


class SemiLinearWendlandKernel(SemiLinearKernelMixin, WendlandKernel):
    def __init__(self, d, power, sigma_max, sigma_min,
                 anisotropic=False, mask=False, D=None):
        WendlandKernel.__init__(
            self,
            d=d,
            power=power,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )
        self._init_semilinear_kernel(mask=mask, D=D)


class SemiLinearMaternKernel(SemiLinearKernelMixin, MaternKernel):
    def __init__(self, d, power, sigma_max, sigma_min,
                 anisotropic=False, mask=False, D=None, nu=1.5):
        MaternKernel.__init__(
            self,
            d=d,
            power=power,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            nu=nu,
        )
        self._init_semilinear_kernel(mask=mask, D=D)


#########################################################
####################### exact solutions ################## 
#########################################################



# sine functions

def f_sines(x):
        x = jnp.atleast_2d(x)  # Ensures x has shape (N, 2)
        result = (2 * jnp.pi**2 * jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1]) + 
                 16 * jnp.pi**2 * jnp.sin(2 * jnp.pi * x[:, 0]) * jnp.sin(2 * jnp.pi * x[:, 1]))
        result += ex_sol_sines(x) ** 3

        return result if x.shape[0] > 1 else result[0]  # Return scalar if input was (2,)

def ex_sol_sines(x):
    x = jnp.atleast_2d(x)  # Ensures x has shape (N, 2)
    result = (jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1]) +
            2*jnp.sin(2*jnp.pi * x[:, 0]) * jnp.sin(2 * jnp.pi * x[:, 1]))
    return result if x.shape[0] > 1 else result[0]  # Return scalar if input was (2,)




# two bump functions
def ex_sol_help_bump(x, center=(0.30, 0.30), k=8, R_0=0.2):
    x = jnp.atleast_2d(x)  # Ensures x has shape (N, 2)
    R = jnp.sqrt((x[:, 0] - center[0])**2 + (x[:, 1] - center[1])**2)
    return jnp.tanh(k * (R_0 - R)) + 1

def f_help_bump(x, center=(0.2, 0.30), k=8, R_0=0.2):
    x = jnp.atleast_2d(x)  # Ensures x has shape (N, 2)
    R = jnp.sqrt((x[:, 0] - center[0])**2 + (x[:, 1] - center[1])**2)
    tanh_term = jnp.tanh(k * (R_0 - R))
    tanh_sq = tanh_term**2
    term_x = (-2 * k * (x[:, 0] - center[0])**2 * tanh_term / R**2) + ((x[:, 0] - center[0])**2 / R**3) - (1 / R)
    term_y = (-2 * k * (x[:, 1] - center[1])**2 * tanh_term / R**2) + ((x[:, 1] - center[1])**2 / R**3) - (1 / R)
    result = k * (tanh_sq - 1) * (term_x + term_y)
    return result


R_1 = 0.3
R_2 = 0.15
center_1 = [0.30, 0.30]
center_2 = [-0.30, -0.30]
k1 = 12
k2 = 4

def ex_sol_bump(x):
    return ex_sol_help_bump(x, center=center_1, k=k1, R_0=R_1) + ex_sol_help_bump(x, center=center_2, k=k2, R_0=R_2)


def f_bump(x):
    return f_help_bump(x, center=center_1, k=k1, R_0=R_1) + f_help_bump(x, center=center_2, k=k2, R_0=R_2) + ex_sol_bump(x) ** 3




    
class PDE:
    KERNEL_REGISTRY: Dict[str, Callable[["PDE"], Any]] = {
        "gaussian": lambda kcfg, p: SemiLinearGaussianKernel(
            d=p.d,
            D=p.D,
            power=p.power,
            sigma_max=kcfg.get('sigma_max', 1),
            sigma_min=kcfg.get('sigma_min', 1e-3),
            anisotropic=kcfg.get('anisotropic', False),
            mask=p.mask,
        ),
        "wendland": lambda kcfg, p: SemiLinearWendlandKernel(
            d=p.d,
            D=p.D,
            power=p.power,
            sigma_max=kcfg.get('sigma_max', 1),
            sigma_min=kcfg.get('sigma_min', 1e-3),
            mask=p.mask,
        ),
        "matern32": lambda kcfg, p: SemiLinearMaternKernel(
            d=p.d,
            D=p.D,
            power=p.power,
            sigma_max=kcfg.get('sigma_max', 1),
            sigma_min=kcfg.get('sigma_min', 1e-3),
            mask=p.mask,
            nu=kcfg.get('nu', 1.5),
        ),
        }
    
    EXACT_SOL_REGISTRY = {
            "sines":{
                'f': f_sines,
                'ex_sol': ex_sol_sines
            },
            
            "two_bumps": {
                'f': f_bump,
                'ex_sol': ex_sol_bump
            }
        }
    
    def __init__(self, pcfg, kcfg):
        """
        Initializes the problem setup for a neural network-based Laplacian solver.
        """
        # Problem parameters
        self.name = 'SemiLinearPDE'
        self.power=pcfg.get('power', 4.01)

        self.d = 2  # spatial dimension
        
        
        self.scale = pcfg.get('scale', 1.0) 
        self.mask = (self.scale < 1e-5)
        print(f"SemiLinearPDE: using mask = {self.mask} based on scale = {self.scale}")
        self.seed = pcfg.get('seed', 200)
        self.key = jax.random.PRNGKey(self.seed)


        # domain for the input weights
        self.D = jnp.array([
                [-1., 1.],
                [-1., 1.],
        ])

        self.vol_D = jnp.prod(self.D[:, 1] - self.D[:, 0])


        self.kernel = self._build_kernel(kcfg)
        self.init_pad_size = pcfg.get('init_pad_size', 16)
         
        if kcfg.get('anisotropic', False):
            self.dim = 2 * self.d # weight dimension
        else:
            self.dim = self.d + 1


        self.Omega = jnp.array([
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-10.0, 0.0],
        ])
        
        if kcfg.get('anisotropic', False):
            self.Omega = jnp.vstack([self.Omega[:self.d, :], jnp.tile(self.Omega[self.d, :], (self.d, 1))])

        assert self.dim == self.Omega.shape[0] and self.d == self.Omega.shape[1]


        self.u_zero = {"x": jnp.zeros((self.init_pad_size, self.d)), 
                       "s": jnp.zeros((self.init_pad_size, self.dim-self.d)),  
                       "u": jnp.zeros((self.init_pad_size))} 
        # Observation set
        self.Nobs = pcfg.get('Nobs', 50)

        self.xhat_int, self.xhat_bnd = self.sample_obs(self.Nobs, method=pcfg.get('sampling', 'grid'))
        self.xhat = jnp.vstack([self.xhat_int, self.xhat_bnd])
        self.Nx_int = self.xhat_int.shape[0]
        self.Nx_bnd = self.xhat_bnd.shape[0]
        self.Nx = self.Nx_int + self.Nx_bnd
        # Optimization-related attributes
        self.obj = Objective(self.Nx_int, self.Nx_bnd, scale=self.scale)
        
        self.Ntest = 100
        self.test_int, self.test_bnd = self.sample_obs(self.Ntest, method='grid')
        self.obj_test = Objective(self.test_int.shape[0], self.test_bnd.shape[0], scale=self.scale)


        # exact solution and rhs
        self.rhs_type = pcfg.get('rhs_type', 'sines')
        self._build_exact_sol_rhs(self.rhs_type)

    def _build_kernel(self, kcfg):
        kernel_type = kcfg.get('type', 'gaussian')
        if kernel_type not in self.KERNEL_REGISTRY:
            raise ValueError(
                f"Unknown kernel_type '{kernel_type}' for SemiLinearPDE. "
                f"Available: {list(self.KERNEL_REGISTRY.keys())}"
            )
        builder = self.KERNEL_REGISTRY[kernel_type]
        return builder(kcfg, self)

    def _build_exact_sol_rhs(self, ex_sol_key):
        if ex_sol_key in self.EXACT_SOL_REGISTRY:
            self.f = self.EXACT_SOL_REGISTRY[ex_sol_key]['f']
            self.ex_sol = self.EXACT_SOL_REGISTRY[ex_sol_key]['ex_sol']
        else:
            raise ValueError(f"Unknown exact_solution key '{ex_sol_key}'. Available: {list(self.EXACT_SOL_REGISTRY.keys())}")

    
    def sample_obs(self, Nobs, method='grid'):
        """
        Samples observations from D
        method: 'uniform' or 'grid'
        """

        # obs_int, obs_bnd = sample_cube_obs(self.D, Nobs, method=method)
        Nobs_int, Nobs_bnd = int((Nobs - 2)**2), 4 * (Nobs - 1)
        obs_int, obs_bnd = sample_cube_obs(self.D, Nobs_int, Nobs_bnd, method=method, rng=self.key)
        return obs_int, obs_bnd

    def sample_param(self, Ntarget):
        """
        Generates Ntarget random parameters in the desired parameter set.
        """
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)

        randomx = self.Omega[:self.d, 0] + (self.Omega[:self.d, 1] - self.Omega[:self.d, 0]) * \
                  jax.random.uniform(subkey1, shape=(Ntarget, self.d))

        randoms = self.Omega[-1, 0] + (self.Omega[self.d:, 1] - self.Omega[self.d:, 0]) * \
                  jax.random.uniform(subkey2, shape=(Ntarget, self.dim - self.d))


        return randomx, randoms

    def plot_forward(self, x, s, c, suppc=None):
        """
        Plots the forward solution.
        """
        plot_solution_2d(self, x, s, c, suppc=None)
        


