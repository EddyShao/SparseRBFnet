# import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.kernel.Kernels import GaussianKernel2DAnisotropic as GaussianKernel
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

    def _init_semilinear_kernel(self, mask=False, D=None, epsilon=0.1, L=None):
        """Call this in subclass __init__ after base-kernel init."""
        self.mask = mask
        self.D = D
        if L is not None:
            self.L = L
        else:   
            self.L = jnp.eye(self.D.shape[0])
            

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
        return self.L.T @ jax.grad(self.kappa_X_c, argnums=3)(X, S, c, xhat)
    
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
        nabla_v = self.L.T @ jax.grad(self.kappa, argnums=2)(x, s, xhat)
        lap_v = jnp.trace(jax.hessian(self.kappa, argnums=2)(x, s, xhat))
        return 2*jnp.dot(nabla_u, nabla_v) - self.epsilon*lap_v

    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa(self, x, s, xhat, *args):
        return self.kappa(x, s, xhat)


class EikonalGaussianKernel(EikonalKernelMixin, GaussianKernel):
    def __init__(self, d, power, sigma_max, sigma_min,
                 anisotropic=False, mask=False, D=None, epsilon=0.1, L=None):
        GaussianKernel.__init__(
            self,
            d=d,
            power=power,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            anisotropic=anisotropic,
        )
        self._init_semilinear_kernel(mask=mask, D=D, epsilon=epsilon, L=L)

def dist_M_to_boundary(x, M):
    # x: (..., 2)
    a = M[0, 0]
    b = M[0, 1]
    c = M[1, 1]

    x1, x2 = x[..., 0], x[..., 1]

    def quad_norm(x, y):
        diff = x - y
        return jnp.einsum('...i,ij,...j->...', diff, M, diff)

    # right side
    t1 = x2 + (b / c) * (x1 - 1.0)
    t1c = jnp.clip(t1, -1.0, 1.0)
    y1 = jnp.stack([jnp.ones_like(x1), t1c], axis=-1)
    d1_sq = quad_norm(x, y1)

    # left side
    t2 = x2 + (b / c) * (x1 + 1.0)
    t2c = jnp.clip(t2, -1.0, 1.0)
    y2 = jnp.stack([-jnp.ones_like(x1), t2c], axis=-1)
    d2_sq = quad_norm(x, y2)

    # top side
    s3 = x1 + (b / a) * (x2 - 1.0)
    s3c = jnp.clip(s3, -1.0, 1.0)
    y3 = jnp.stack([s3c, jnp.ones_like(x2)], axis=-1)
    d3_sq = quad_norm(x, y3)

    # bottom side
    s4 = x1 + (b / a) * (x2 + 1.0)
    s4c = jnp.clip(s4, -1.0, 1.0)
    y4 = jnp.stack([s4c, -jnp.ones_like(x2)], axis=-1)
    d4_sq = quad_norm(x, y4)

    d_sq = jnp.minimum(jnp.minimum(d1_sq, d2_sq), jnp.minimum(d3_sq, d4_sq))
    return jnp.sqrt(d_sq)

    
class PDE:
    KERNEL_REGISTRY: Dict[str, Callable[["PDE"], Any]] = {
        "gaussian": lambda kcfg, p: EikonalGaussianKernel(
            d=p.d,
            D=p.D,
            L=p.L,
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
            "ex_sol": lambda x, L: dist_M_to_boundary(x, jnp.linalg.inv(L @ L.T)),
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

        self.L = jnp.array([
                [2.6,             0.       ],
                [-4.0,             1.8],
            ])

        self.kernel = self._build_kernel(kcfg)
        self.init_pad_size = pcfg.get('init_pad_size', 16)
         
        if not kcfg.get('anisotropic', False):
            raise ValueError("EikonalAnisoAniso.py requires 'anisotropic' = True in kernel config.")


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
        
        

        assert self.dim == self.Omega.shape[0] 

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
            self.ex_sol = lambda x: self.EXACT_SOL_REGISTRY[ex_sol_key]['ex_sol'](x, self.L)
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
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)

        # centers x: (N,2)
        lo_x = self.Omega[:self.d, 0]
        hi_x = self.Omega[:self.d, 1]
        randomx = lo_x + (hi_x - lo_x) * jax.random.uniform(subkey1, (Ntarget, self.d))

        # anisotropic params s = (theta, r1, r2): (N,3)
        lo_s = self.Omega[self.d:, 0]
        hi_s = self.Omega[self.d:, 1]
        randoms = lo_s + (hi_s - lo_s) * jax.random.uniform(subkey2, (Ntarget, self.dim - self.d))

        return randomx, randoms

    def plot_forward(self, x, s, c, suppc=None):
        """
        Plots the forward solution.
        """
        if suppc is None:
            suppc = np.ones_like(c, dtype=bool)
        # assert self.dim == 3 

        # # Extract the domain range
        # pO = self.Omega[:-1, :]
        plt.close('all')  # Close previous figure to prevent multiple windows

        # Create a new figure
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133)

        t_x = np.linspace(self.D[0, 0], self.D[0, 1], 100)
        t_y = np.linspace(self.D[1, 0], self.D[1, 1], 100)
        X, Y = np.meshgrid(t_x, t_y)
        t = np.vstack((X.flatten(), Y.flatten())).T

        if self.ex_sol is not None:
            f1 = self.ex_sol(t).reshape(X.shape)
        # Plot exact solution
        surf1 = ax1.plot_surface(X, Y, f1, cmap='viridis', edgecolor='none')
        ax1.set_title("Exact Solution")
        ax1.set_xlabel("X-axis")
        ax1.set_ylabel("Y-axis")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

        # Compute predicted solution
        Gu = self.kernel.kappa_X_c_Xhat(x, s, c, t)
        # sigma is sigmoid of S

        # Plot predicted solution
        surf2 = ax2.plot_surface(X, Y, Gu.reshape(X.shape), cmap='viridis', edgecolor='none')
        ax2.set_title("Predicted Solution") 
        ax2.set_xlabel("X-axis")
        ax2.set_ylabel("Y-axis")
        ax2.set_zlabel("$f_2(x, y)$")
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)


        # plot all collocation point X
        # together with error countour plot
        contour = ax3.contourf(X, Y, np.abs(Gu.reshape(100, 100) - f1), cmap='viridis')        
        # ax3.scatter(x[:, 0].flatten(), x[:, 1].flatten(), color='r', marker='x')
        if self.anisotropic:
            # Get per-center R (shape: (N, 2, 2)); convert to numpy for matplotlib
            
            for i, (xi, yi) in enumerate(x[:, :2]):
                if suppc is not None and not bool(suppc[i]):
                    continue
                # Extract the i-th R matrix
                s_i = s[i]
                r1 = self.kernel.r_min[0] + (self.kernel.r_max[0] - self.kernel.r_min[0]) * jax.nn.sigmoid(s_i[1])
                r2 = self.kernel.r_min[1] + (self.kernel.r_max[1] - self.kernel.r_min[1]) * jax.nn.sigmoid(s_i[2])
                a1, a2 = 1.0 / r1, 1.0 / r2  # Semi-major and semi-minor axes lengths

                # Rotation angle in degrees from x-axis
                angle_deg = -np.degrees(jax.nn.sigmoid(s_i[0]))  # Map to [0, 90]

                # Draw ellipse and center
                ell = patches.Ellipse((xi, yi),
                                    width=2*a1, height=2*a2, angle=angle_deg,
                                    edgecolor='r', facecolor='none',
                                    linestyle='dashed', linewidth=1, label="Reference ellipse")
                ax3.add_patch(ell)
                ax3.scatter(xi, yi, color='r', marker='x')
        else:
            raise NotImplementedError("only handles anisotropic case")

        ax3.set_aspect('equal')  # Ensures circles are properly shaped
        # # set colorbars
        ax3.set_xlim(self.Omega[0, 0], self.Omega[0, 1])
        ax3.set_ylim(self.Omega[1, 0], self.Omega[1, 1])
        ax3.set_title("Collocation Points, Error Contour") 
        fig.colorbar(contour, ax=ax3, shrink=0.5, aspect=5)   

        plt.show(block=False)
        plt.pause(1.0)  
        




    

