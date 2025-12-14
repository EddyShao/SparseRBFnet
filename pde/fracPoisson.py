# import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.kernel.Kernels import GaussianKernel
from src.utils import Objective, sample_cube_obs
# from src.fracLapRBF import FractionalLaplacianRBF
from src.frac.build_interp import make_interp1d_with_custom_deriv
import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import gamma
from scipy.special import hyp2f1, gammaln

# disable jit for debugging
# jax.config.update("jax_disable_jit", True)

def _polar_to_euclid(mesh):
    """mesh: (M, 2) with columns [r, theta]"""
    r = mesh[:, 0]
    th = mesh[:, 1]
    x = r * jnp.cos(th)
    y = r * jnp.sin(th)
    return jnp.stack([x, y], axis=1)

def _spherical_to_euclid(mesh):
    """
    mesh: (M, 3) with columns [r, phi, theta]
      phi   = azimuth in x-y plane  (e.g. [-pi, pi) )
      theta = polar/colatitude from +z (e.g. [0, pi) )
    """
    r = mesh[:, 0]
    phi = mesh[:, 1]
    th = mesh[:, 2]
    x = r * jnp.sin(th) * jnp.cos(phi)
    y = r * jnp.sin(th) * jnp.sin(phi)
    z = r * jnp.cos(th)
    return jnp.stack([x, y, z], axis=1)


data = jnp.load("fracLapRBF_d_3_frac_order_1_gaussian.npz")
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
            r = jnp.linalg.norm(x - xhat)
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

def ex_sol(x, frac_order=1., d=3):
    """
    Exact solution for the restricted/integral fractional Laplacian on the unit ball:

        u(x) = C_{d,alpha} * (1 - ||x||^2)_+^{alpha/2}

    Parameters
    ----------
    x : array_like, shape (d,) or (N,d)
        Spatial points
    frac_order : float
        alpha in (0,2]
    d : int
        Spatial dimension

    Returns
    -------
    u : array
        Values of u at x
    """
    alpha = float(frac_order)
    d = int(d)

    x = jnp.asarray(x)
    x = jnp.atleast_2d(x)            # (N,d)

    r2 = jnp.sum(x * x, axis=1)
    inside = 1.0 - r2

    # C_{d,alpha} = 2^{-alpha} * Gamma(d/2)
    #               / (Gamma((d+alpha)/2) * Gamma(1+alpha/2))
    logC = (-alpha * jnp.log(2.0)
            + gammaln(0.5 * d)
            - gammaln(0.5 * (d + alpha))
            - gammaln(1.0 + 0.5 * alpha))
    C = jnp.exp(logC)

    u = C * jnp.where(inside > 0.0, inside ** (0.5 * alpha), 0.0)

    return u if x.shape[0] > 1 else u[0]

def rhs(x):
    """
    RHS for the unit-ball fractional Laplacian test:
        (-Δ)^{α/2} u = 1   in |x|<1
        u = 0              outside

    Parameters
    ----------
    x : array_like, shape (d,) or (N,d)
    frac_order : float
        (unused here, included for interface consistency)
    d : int
        (unused here, included for interface consistency)

    Returns
    -------
    f : array
        RHS values
    """
    x = jnp.asarray(x)
    x = jnp.atleast_2d(x)

    r2 = jnp.sum(x * x, axis=1)
    f = jnp.where(r2 < 1.0, 1.0, 0.0)

    return f if x.shape[0] > 1 else f[0]

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

    
    EXACT_SOL_REGISTRY = {
        "one": {
            "f": rhs,
            "ex_sol": lambda x, frac_order=1., d=1: ex_sol(x, frac_order, d),
        },
    }
    

    
    def __init__(self, pcfg: dict, kcfg: dict):
        """
        Initializes the problem setup for a neural network-based Laplacian solver.
        """
        # Problem parameters
        self.name = 'fracPoisson'
        self.d = int(pcfg.get("d", 4))
        self.frac_order = pcfg.get('frac_order', 1.0)
        self.power = float(pcfg.get("power", self.d + self.frac_order + 0.01))
        self.scale = float(pcfg.get("scale", 1.0))

        self.seed = int(pcfg.get("seed", 200))
        self.key = jax.random.PRNGKey(self.seed)


        # domain for the input weights

        if self.d == 1:
            self.D = jnp.array([[-1.0, 1.0]])
            self.vol_D = 2.0
        elif self.d == 2:
            self.D = jnp.array([[0, 1.0],
                                [-jnp.pi, jnp.pi]])
            self.vol_D = jnp.pi
        elif self.d == 3:
            self.D = jnp.array([[0, 1.0],
                                [-jnp.pi, jnp.pi],
                                [0, jnp.pi]])
            self.vol_D = 4 * jnp.pi / 3

        else:
            raise NotImplementedError("Domain D not implemented for d > 3")

        self.CompBox = jnp.vstack([jnp.array([-1., 1.]) for _ in range(self.d)])  # computational box
        

        self.anisotropic = kcfg.get('anisotropic', False)
        
        print('Fractional order:', self.frac_order)

        self.kernel = self._build_kernel(kcfg)
        
        if self.anisotropic:
            self.dim = 2 * self.d # weight dimension
        else:
            self.dim = self.d + 1


        # copy jnp.D to self.Omega
         # Domain for sampling input weights
        Omega_x = self.D
        Omega_x = Omega_x.at[0, :].set(2.0 * Omega_x[0, :])
        Omega_s = jnp.array([[-10, 0.0]])
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
        print(self.xhat_int)
        self.Nx_int = self.xhat_int.shape[0]
        self.Nx_bnd = self.xhat_bnd.shape[0]
        self.Nx = self.Nx_int + self.Nx_bnd
        # Optimization-related attributes
        self.obj = Objective(self.Nx_int, self.Nx_bnd, scale=self.scale)

        self.Ntest = 200
        self.test_int, self.test_bnd = self.sample_obs(self.Ntest, method='grid')
        self.rhs_type = pcfg.get('rhs_type', 'one')
        self._build_exact_sol_rhs(self.rhs_type)

    def _build_kernel(self, kcfg: dict):
        ktype = kcfg.get("type", "gaussian")
        if ktype not in self.KERNEL_REGISTRY:
            raise ValueError(
                f"Unknown kernel_type '{ktype}' for SemiLinearHighDim. "
                f"Available: {list(self.KERNEL_REGISTRY.keys())}"
            )
        builder = self.KERNEL_REGISTRY[ktype]
        return builder(self, kcfg)
    
    def _build_exact_sol_rhs(self, ex_sol_key):
        if ex_sol_key in self.EXACT_SOL_REGISTRY:
            self.f = self.EXACT_SOL_REGISTRY[ex_sol_key]['f']
            self.ex_sol = lambda x: self.EXACT_SOL_REGISTRY[ex_sol_key]['ex_sol'](x, self.frac_order, self.d)
        else:
            raise ValueError(f"Unknown exact_solution key '{ex_sol_key}'. Available: {list(self.EXACT_SOL_REGISTRY.keys())}")

    # NEED TO MODIFY BELOW FOR SPHERICAL CASE

    def sample_test_obs(self, Ntest):
        """
        Sample test interior and boundary/exterior points using a grid method.
        """
        # actually we should do quadrature in a 
        Ntest_int, Ntest_bnd = int((Ntest-2)**self.d), int((Ntest**self.d) - (Ntest-2)**self.d)
        test_int, test_bnd = sample_cube_obs(self.CompBox, Ntest_int, Ntest_bnd, method="grid")
        mask_int = jnp.norm(test_int, axis=1) <= 1.0
        mask_bnd = jnp.norm(test_int, axis=1) > 1.0
        return test_int[mask_int], test_int[mask_bnd]
    

    # NEED TO MODIFY BELOW FOR SPHERICAL CASE
    
    def sample_obs(self, Nobs, method="grid"):
        """
        Grid sample in (r, angles) on self.D, map to Euclidean, then split by r<1.
        Angle axes (ax >= 1) use endpoint=False to avoid duplicating the seam.
        """
        if method != "grid":
            raise NotImplementedError("Only method='grid' is implemented here.")

        # 1) Build coordinate vectors; drop one endpoint for angles (axes >= 1)
        coords_list = []
        for ax in range(self.d):
            a, b = 2 * self.D[ax, 0], 2 * self.D[ax, 1]
            endpoint = (ax == 0)  # r includes endpoint; angles do NOT
            coords = jnp.linspace(a, b, Nobs, endpoint=endpoint)
            coords_list.append(coords)

        # 2) Full tensor grid -> (M, d)
        grids = jnp.meshgrid(*coords_list, indexing="ij")
        mesh = jnp.stack([g.reshape(-1) for g in grids], axis=1)  # (M, d)

        # 3) Split by r < 1 vs r >= 1 (in parameter space)
        r = mesh[:, 0]
        mask_int = jnp.abs(r) < 1.0
        mesh_int = mesh[mask_int]
        mesh_bnd = mesh[~mask_int]

        # 4) Map to Euclidean
        if self.d == 1:
            # If d==1, "r" is just x already (no angle)
            obs_int = mesh_int[:, :1]
            obs_bnd = mesh_bnd[:, :1]
        elif self.d == 2:
            obs_int = _polar_to_euclid(mesh_int)
            obs_bnd = _polar_to_euclid(mesh_bnd)
        elif self.d == 3:
            obs_int = _spherical_to_euclid(mesh_int)
            obs_bnd = _spherical_to_euclid(mesh_bnd)
        else:
            raise ValueError(f"Unsupported d={self.d} for polar/spherical mapping.")

        return obs_int, obs_bnd

# NEED TO MODIFY BELOW FOR SPHERICAL CASE
    def sample_param(self, Ntarget):
        """
        Generates Ntarget random parameters in the desired parameter set.
        """

        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
        randomp = self.Omega[0, 0] + (self.Omega[:self.d, 1] - self.Omega[:self.d, 0]) * jax.random.uniform(
            subkey1, shape=(Ntarget, self.d)
        )
        randoms = self.Omega[-1, 0] + (self.Omega[self.d:, 1] - self.Omega[self.d:, 0]) * jnp.tile(
            jax.random.uniform(subkey2, shape=(Ntarget, 1)), (1, self.dim - self.d)
        )
        if self.d == 1:
            randomx = randomp
        elif self.d == 2:
            randomx = _polar_to_euclid(randomp)
        elif self.d == 3:
            randomx = _spherical_to_euclid(randomp)
        else:
            raise ValueError(f"Unsupported d={self.d} for polar/spherical mapping.")

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




