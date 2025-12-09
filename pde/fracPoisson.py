# import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.Kernels import GaussianKernel
from src.utils import Objective, sample_cube_obs
# from src.fracLapRBF import FractionalLaplacianRBF
from src.build_interp import make_interp1d_with_custom_deriv
import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import gamma
from scipy.special import hyp2f1
jax.config.update("jax_enable_x64", True)

# disable jit for debugging
# jax.config.update("jax_disable_jit", True)
data = jnp.load("/Users/zs/Desktop/SparseKernelPDE/fracLapRBF_example_output.npz")
r = data['r']
y_grid = data['y']
dy_grid = data['dy']

LapFrac = make_interp1d_with_custom_deriv(r, y_grid, dy_grid)

class fracGaussianKernel(GaussianKernel):
    def __init__(self, d, power, sigma_max, sigma_min, frac_order=1.0, anisotropic=False, mask=False, D=None):
        super().__init__(d=d, power=power, sigma_max=sigma_max, sigma_min=sigma_min, anisotropic=anisotropic)
        self.mask = mask
        self.D = D
        self.frac_order = frac_order

        # linear results for computing E and B
        self.linear_E = (self.Lap_kappa_X_c_Xhat,)
        self.linear_B = (self.kappa_X_c_Xhat,)
        self.DE = () 
        self.DB = ()  

    
    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c(self, X, S, c, xhat):
        def Lap_kappa_single(x, s):
            sigma = self.sigma(s)[0]
            eps = 1 / (2 * sigma**2)
            r = jnp.linalg.norm(x - xhat)**2
            scaled_r = r * jnp.sqrt(eps)
            
            coef = (sigma ** self.power) / (
                (jnp.sqrt(2 * jnp.pi) * sigma) ** self.d
            )
            sigma_cor = eps ** (self.frac_order / 2)

            return coef * sigma_cor * LapFrac(scaled_r)
        val = jax.vmap(Lap_kappa_single, in_axes=(0, 0))(X, S)

        return jnp.dot(val.flatten(), c)
    

    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c_Xhat(self, X, S, c, Xhat): 
        return jax.vmap(self.Lap_kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    @partial(jax.jit, static_argnums=(0,))
    def E_kappa_X_c(self, X, S, c, xhat):
        return self.Lap_kappa_X_c(X, S, c, xhat)

    @partial(jax.jit, static_argnums=(0,))
    def B_kappa_X_c(self, X, S, c, xhat):
        return self.kappa_X_c(X, S, c, xhat)
    

    def E_kappa_X_c_Xhat(self, *linear_results):
        return linear_results[0] 

    def B_kappa_X_c_Xhat(self, *linear_results):
        return linear_results[0]

    @partial(jax.jit, static_argnums=(0,))
    def DE_kappa(self, x, s, xhat, *args):
        sigma = self.sigma(s)[0]
        eps = 1 / (2 * sigma**2)
        r = jnp.linalg.norm(x - xhat)
        scaled_r = r * jnp.sqrt(eps)
        
        coef = (sigma ** self.power) / (
            (jnp.sqrt(2 * jnp.pi) * sigma) ** self.d
        )
        sigma_cor = eps ** (self.frac_order / 2)

        return coef * sigma_cor * LapFrac(scaled_r)



    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa(self, x, s, xhat, *args):
        return self.kappa(x, s, xhat)

    
class PDE:



    KERNEL_REGISTRY = {
        "fracGaussian": lambda p, kcfg: fracGaussianKernel(
            d = p.d,
            power = kcfg.get('power', p.d + p.frac_order + 0.01),
            sigma_max=kcfg.get("sigma_max", 1.0),
            sigma_min=kcfg.get("sigma_min", 1e-3),
            anisotropic=kcfg.get("anisotropic", False),
            frac_order = p.frac_order,
            D=p.D
        ),
    }

    
    def __init__(self, pcfg: dict, kcfg: dict):
        """
        Initializes the problem setup for a neural network-based Laplacian solver.
        """
        # Problem parameters
        self.name = 'fracPoisson'
        self.d = int(pcfg.get("d", 4))
        self.power = float(pcfg.get("power", self.d + 2.01))
        self.scale = float(pcfg.get("scale", 1.0))

        self.seed = int(pcfg.get("seed", 200))
        self.key = jax.random.PRNGKey(self.seed)


        # domain for the input weights
        self.D = jnp.stack([
                        jnp.array([-1., 1.])
                        for _ in range(self.d)
                    ])

        self.vol_D = jnp.prod(self.D[:, 1] - self.D[:, 0])

        self.anisotropic = kcfg.get('anisotropic', False)
        self.frac_order = pcfg.get('frac_order', 1.0)
        print('Fractional order:', self.frac_order)

        self.kernel = self._build_kernel(kcfg)
        
        if self.anisotropic:
            self.dim = 2 * self.d # weight dimension
        else:
            self.dim = self.d + 1


        # x in [-2, 2]^d
        Omega_x = jnp.tile(jnp.array([[-2.0, 2.0]]), (self.d, 1))
        # s scalar in [-10, 0]
        Omega_s = jnp.array([[-10.0, 0.0]])
        self.Omega = jnp.vstack([Omega_x, Omega_s])
        
        if self.anisotropic:
            self.Omega = jnp.vstack([self.Omega[:self.d, :], jnp.tile(self.Omega[self.d, :], (self.d, 1))])


        self.pad_size = 16
        self.u_zero = {"x": jnp.zeros((self.pad_size, self.d)),
                       "s": jnp.zeros((self.pad_size, self.dim-self.d)),  
                       "u": jnp.zeros((self.pad_size))} # initial solution for anisotropic


        # Observation set
        self.Nobs = pcfg.get('Nobs', 50)
        self.method = pcfg.get('sampling', 'grid')

        self.xhat_int, self.xhat_bnd = self.sample_obs(self.Nobs, method=self.method)
        self.xhat = jnp.vstack([self.xhat_int, self.xhat_bnd])
        self.Nx_int = self.xhat_int.shape[0]
        self.Nx_bnd = self.xhat_bnd.shape[0]
        self.Nx = self.Nx_int + self.Nx_bnd
        # Optimization-related attributes
        self.obj = Objective(self.Nx_int, self.Nx_bnd, scale=self.scale)
        self.Ntest = 200

        self.test_int, self.test_bnd = self.sample_obs(self.Ntest, method='grid')

    def _build_kernel(self, kcfg: dict):
        ktype = kcfg.get("type", "gaussian")
        if ktype not in self.KERNEL_REGISTRY:
            raise ValueError(
                f"Unknown kernel_type '{ktype}' for SemiLinearHighDim. "
                f"Available: {list(self.KERNEL_REGISTRY.keys())}"
            )
        builder = self.KERNEL_REGISTRY[ktype]
        return builder(kcfg, self)
    
    # NEED TO MODIFY BELOW FOR SPHERICAL CASE

    def sample_test_obs(self, Ntest):
        """
        Sample test interior and boundary/exterior points using a grid method.
        """
        return sample_cube_obs(Ntest, method="grid")
    

    # NEED TO MODIFY BELOW FOR SPHERICAL CASE
    
    def sample_obs(self, Nobs, method="grid"):
        """
        Sample interior and boundary/exterior points.

        - if method == "frac_grid": build a full grid on [-2,2]^d,
        then classify points inside [-1,1]^d vs outside.
        - otherwise: fall back to the standard cube sampling.
        """
        if method == 'frac_grid':
            # --------- new fractional collocation: full grid on [-2,2]^d ---------
            Ngrid_per_dim = int(jnp.asarray(
                jnp.array(Nobs)
            ))  # or pass pcfg into PDE, or store beforehand

            # 1D grid in [-2,2]
            grid_1d = jnp.linspace(-2.0, 2.0, Ngrid_per_dim)  # (Ngrid_per_dim,)

            # Cartesian product -> shape (Ngrid_per_dim^d, d)
            meshes = jnp.meshgrid(*[grid_1d] * self.d, indexing="ij")
            grid = jnp.stack(meshes, axis=-1).reshape(-1, self.d)

            # interior mask: all coords in [-1,1]
            interior_mask = jnp.logical_and(
                jnp.all(grid >= -1.0, axis=1),
                jnp.all(grid <=  1.0, axis=1),
            )

            obs_int = grid[interior_mask]       # points inside physical domain
            obs_bnd = grid[~interior_mask]      # "exterior" points

            # If you *really* want to enforce Nobs_int / Nobs_bnd,
            # you can slice here (e.g. obs_int[:Nobs_int], etc.)
            return obs_int, obs_bnd
        
        elif method == "bnd_emphasis":
            # --------- continuous double-Gaussian boundary emphasis ---------
            Ngrid_1d = int(Nobs)
            bnd_sigma = float(self.pcfg.get("bnd_sigma", 0.15))   # width of peaks
            w     = float(self.pcfg.get("bnd_weight", 0.45))  # peak weight
            
            # 1D reference mesh
            x_ref = jnp.linspace(-2.0, 2.0, 20001)
            
            # double-Gaussian PDF
            def pdf(x):
                ga1 = jnp.exp(-((x + 1.0)**2) / (2*bnd_sigma**2))
                ga2 = jnp.exp(-((x - 1.0)**2) / (2*bnd_sigma**2))
                p = w * ga1 + w * ga2 + (1 - 2*w) * 1.0      # uniform background
                return p
            
            p_ref = pdf(x_ref)
            p_ref = p_ref / jnp.trapz(p_ref, x_ref)         # normalize
            
            cdf_ref = jnp.cumsum(p_ref)
            cdf_ref = cdf_ref / cdf_ref[-1]
            
            # invert CDF on uniform grid
            u = jnp.linspace(0.0, 1.0, Ngrid_1d)
            grid_1d = jnp.interp(u, cdf_ref, x_ref)
            
            # Cartesian grid in d dimensions
            meshes = jnp.meshgrid(*[grid_1d]*self.d, indexing="ij")
            grid   = jnp.stack(meshes, axis=-1).reshape(-1, self.d)
            
            # classify points
            interior_mask = jnp.logical_and(
                jnp.all(grid >= -1.0, axis=1),
                jnp.all(grid <=  1.0, axis=1),
            )
            
            obs_int = grid[interior_mask]
            obs_bnd = grid[~interior_mask]
            
            return obs_int, obs_bnd

# NEED TO MODIFY BELOW FOR SPHERICAL CASE
    def sample_param(self, Ntarget):
        """
        Generates Ntarget random parameters in the desired parameter set.
        """

        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
        randomx = self.Omega[0, 0] + (self.Omega[:self.d, 1] - self.Omega[:self.d, 0]) * jax.random.uniform(
            subkey1, shape=(Ntarget, self.d)
        )
        randoms = self.Omega[-1, 0] + (self.Omega[self.d:, 1] - self.Omega[self.d:, 0]) * jnp.tile(
            jax.random.uniform(subkey2, shape=(Ntarget, 1)), (1, self.dim - self.d)
        )

        return randomx, randoms

    def plot_forward(self, x, s, c, suppc):
        """
        Plots the forward solution.
        """
        # assert self.dim == 3 

        # # Extract the domain range
        # pO = self.Omega[:-1, :]
        plt.close('all')  # Close previous figure to prevent multiple windows

        # Create a new figure
        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(111)
        t_x = np.linspace(self.Omega[0, 0]-1, self.Omega[0, 1]+1, 200)
        # extend this to d-dimensions, by adding d - 1 zeros
        t = np.zeros((200, self.d))
        t[:, 0] = t_x

        f1 = self.ex_sol(t)
        # Plot exact solution

        ax1.plot(t_x, f1, label="Exact Solution")
    
        # Compute predicted solution
        Gu = self.kernel.kappa_X_c_Xhat(x, s, c, t)
        # sigma is sigmoid of S
        ax1.plot(t_x, Gu, label="Predicted Solution")
        sigma = self.kernel.sigma(s).flatten()
        for i in range(x.shape[0]):
            if suppc[i]:
                y_i = self.kernel.kappa_X_c_Xhat(x, s, c, x[i:i+1, :])
                plt.scatter(x[i], y_i, color='red', s=sigma[i]*300, marker='x')
        plt.legend()    
        plt.show(block=False)
        plt.pause(1.0)  




