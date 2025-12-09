# import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.kernel.Kernels import GaussianKernel, WendlandKernel, MaternKernel
from src.utils import Objective, sample_cube_obs, plot_solution_2d


from typing import Dict, Callable, Any
import jax
import jax.numpy as jnp
from functools import partial

# set the random seed

class EikonalKernelMixin:
    """
    Mixin with PDE-specific structure for the semilinear equation:
        E(u) = -\Delta u + u^3
    """

    def _init_semilinear_kernel(self, mask=False, D=None, epsilon=0.1):
        """Call this in subclass __init__ after base-kernel init."""
        self.mask = mask
        self.D = D

        # linear results for computing E and B
        self.linear_E = (self.Nabla_kappa_X_c_Xhat, self.Lap_kappa_X_c_Xhat)
        self.linear_B= (self.kappa_X_c_Xhat,)

        # linear results required for computing linearized E and B
        self.DE = (0,) 
        self.DB = ()
        self.epsilon = epsilon

    # ------------ kappa with optional mask ------------

    @partial(jax.jit, static_argnums=(0,))
    def kappa(self, x, s, xhat):
        output = super().kappa(x, s, xhat)
        if self.mask:
            mask = jnp.prod(xhat - self.D[:, 0]) * jnp.prod(self.D[:, 1] - xhat)
            output = output * mask
        return output
    

    @partial(jax.jit, static_argnums=(0,))
    def Nabla_kappa_X_c(self, X, S, c, xhat):
        return jax.grad(self.kappa_X_c, argnums=3)(X, S, c, xhat)
    
    @partial(jax.jit, static_argnums=(0,))
    def Nabla_kappa_X_c_Xhat(self, X, S, c, Xhat): 
        return jax.vmap(self.Nabla_kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    
    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c(self, X, S, c, xhat):
        return jnp.trace(jax.hessian(self.kappa_X_c, argnums=3)(X, S, c, xhat))
    
    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c_Xhat(self, X, S, c, Xhat): 
        return jax.vmap(self.Lap_kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    @partial(jax.jit, static_argnums=(0,))
    def E_kappa_X_c(self, X, S, c, xhat):
        nabla = self.Nabla_kappa_X_c(X, S, c, xhat)
        lap = self.Lap_kappa_X_c(X, S, c, xhat)

        return jnp.dot(nabla, nabla) - self.epsilon * lap 
    
    @partial(jax.jit, static_argnums=(0,))
    def B_kappa_X_c(self, X, S, c, xhat):
        return self.kappa_X_c(X, S, c, xhat)
    

    def E_kappa_X_c_Xhat(self, *linear_results):
        nabla = linear_results[0]
        lap = linear_results[1]
        return (nabla ** 2).sum(axis=1)  - self.epsilon * lap

    def B_kappa_X_c_Xhat(self, *linear_results):
        return linear_results[0]
    
    @partial(jax.jit, static_argnums=(0,))
    def DE_kappa(self, x, s, xhat, *args):
        nabla_u = args[0]
        nabla_v = jax.grad(self.kappa, argnums=2)(x, s, xhat)
        lap_v = jnp.trace(jax.hessian(self.kappa, argnums=2)(x, s, xhat))
        return 2*jnp.dot(nabla_u, nabla_v) - self.epsilon*lap_v

    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa(self, x, s, xhat, *args):
        return self.kappa(x, s, xhat)


class EikonalGaussianKernel(EikonalKernelMixin, GaussianKernel):
    def __init__(self, d, power, sigma_max, sigma_min,
                 anisotropic=False, mask=False, D=None, epsilon=0.1):
        GaussianKernel.__init__(
            self,
            d=d,
            power=power,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            anisotropic=anisotropic,
        )
        self._init_semilinear_kernel(mask=mask, D=D, epsilon=epsilon)


    

    
class PDE:
    KERNEL_REGISTRY: Dict[str, Callable[["PDE"], Any]] = {
        "gaussian": lambda kcfg, p: EikonalGaussianKernel(
            d=p.d,
            D=p.D,
            power=p.power,
            sigma_max=kcfg.get('sigma_max', 1),
            sigma_min=kcfg.get('sigma_min', 1e-3),
            anisotropic=kcfg.get('anisotropic', False),
            mask=p.mask,
            epsilon=p.epsilon
        ),
        # "wendland": lambda kcfg, p: SemiLinearWendlandKernel(
        #     d=p.d,
        #     D=p.D,
        #     power=p.power,
        #     sigma_max=kcfg.get('sigma_max', 1),
        #     sigma_min=kcfg.get('sigma_min', 1e-3),
        #     mask=p.mask,
        # ),
        # "matern32": lambda kcfg, p: SemiLinearMaternKernel(
        #     d=p.d,
        #     D=p.D,
        #     power=p.power,
        #     sigma_max=kcfg.get('sigma_max', 1),
        #     sigma_min=kcfg.get('sigma_min', 1e-3),
        #     mask=p.mask,
        #     nu=kcfg.get('nu', 1.5),
        # ),
        }
    EXACT_SOL_REGISTRY: Dict[str, Dict[str, Callable]] = {
        "ones": {
            "f": lambda x: jnp.ones(x.shape[0]),
            "ex_sol": lambda x: jnp.min(1 - jnp.abs(x), axis=1),
        },
    }

    def __init__(self, pcfg, kcfg):
        """
        Initializes the problem setup for a neural network-based Laplacian solver.
        """
        # Problem parameters
        self.name = 'Eikonal'
        self.power=pcfg.get('power', 4.01)
        self.epsilon = pcfg.get('epsilon', 0.1) # diffusion coefficient

        self.d = pcfg.get('d', 2)  # spatial dimension
        
        
        self.scale = pcfg.get('scale', 1.0) 
        self.mask = (self.scale < 1e-5)
        print(f"Eikonal: using mask = {self.mask} based on scale = {self.scale}")
        self.seed = pcfg.get('seed', 200)
        self.key = jax.random.PRNGKey(self.seed)

        # domain for the input weights
        # self.D = jnp.array([
        #         [-1., 1.],
        #         [-1., 1.],
        # ])
        self.D = jnp.stack([jnp.array([-1.0, 1.0]) for _ in range(self.d)])
        self.vol_D = jnp.prod(self.D[:, 1] - self.D[:, 0])


        self.kernel = self._build_kernel(kcfg)
        self.init_pad_size = pcfg.get('init_pad_size', 16)
         
        if kcfg.get('anisotropic', False):
            self.dim = 2 * self.d # weight dimension
        else:
            self.dim = self.d + 1


        # self.Omega = jnp.array([
        #     [-2.0, 2.0],
        #     [-2.0, 2.0],
        #     [-10.0, 0.0],
        # ])
        self.Omega = jnp.vstack([jnp.array([-2.0, 2.0]) for _ in range(self.d)] + [jnp.array([-10.0, 0.0])])
        
        if kcfg.get('anisotropic', False):
            self.Omega = jnp.vstack([self.Omega[:self.d, :], jnp.tile(self.Omega[self.d, :], (self.d, 1))])

        assert self.dim == self.Omega.shape[0] 


        self.u_zero = {"x": jnp.zeros((self.init_pad_size, self.d)), 
                       "s": jnp.zeros((self.init_pad_size, self.dim-self.d)),  
                       "u": jnp.zeros((self.init_pad_size))} 
        # Observation set
        self.Nobs_int = pcfg.get('Nobs_int', 28 ** 2)
        self.Nobs_bnd = pcfg.get('Nobs_bnd', 30 ** 2 - self.Nobs_int) 

        self.xhat_int, self.xhat_bnd = self.sample_obs(self.Nobs_int, self.Nobs_bnd, method=pcfg.get('sampling', 'grid'))
        self.xhat = jnp.vstack([self.xhat_int, self.xhat_bnd])
        self.Nx_int = self.xhat_int.shape[0]
        self.Nx_bnd = self.xhat_bnd.shape[0]
        self.Nx = self.Nx_int + self.Nx_bnd
        # Optimization-related attributes
        self.obj = Objective(self.Nx_int, self.Nx_bnd, scale=self.scale)
        
        self.Ntest = 100
        self.Ntest_int = int((self.Ntest - 2) ** 2)
        self.Ntest_bnd = self.Ntest ** 2 - self.Ntest_int
        self.test_int, self.test_bnd = self.sample_obs(self.Ntest_int, self.Ntest_bnd, method='grid')
        self.obj_test = Objective(self.test_int.shape[0], self.test_bnd.shape[0], scale=self.scale)


        # exact solution and rhs
        self.rhs_type = pcfg.get('rhs_type', 'ones')
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

    
    def sample_obs(self, Nobs_int, Nobs_bnd, method='grid'):
        """
        Samples observations from D
        method: 'uniform' or 'grid'
        """

        # obs_int, obs_bnd = sample_cube_obs(self.D, Nobs, method=method)
        obs_int, obs_bnd = sample_cube_obs(self.D, Nobs_int, Nobs_bnd, method=method, rng=self.key)
        return obs_int, obs_bnd


    def sample_param(self, Ntarget):
        """
        Generates Ntarget random parameters in the desired parameter set.
        """

        # Suppose key is your PRNGKey
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)

        randomx = self.Omega[:self.d, 0] + (self.Omega[:self.d, 1] - self.Omega[:self.d, 0]) * jax.random.uniform(subkey1, (Ntarget, self.d))
        randoms = self.Omega[-1, 0] + (self.Omega[self.d:, 1] - self.Omega[self.d:, 0]) * jnp.tile(
            jax.random.uniform(subkey2, (Ntarget, 1)), (1, self.dim - self.d)
        )
        return randomx, randoms

    def plot_forward(self, x, s, c, suppc=None):
        if self.d == 2:
            plot_solution_2d(self, x, s, c, suppc=suppc)
        else:
            pass




    

