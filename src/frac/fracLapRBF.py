import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
jax.config.update("jax_enable_x64", True)
# -------------------------- helpers: Bessel J_{±1/2} and derivatives --------------------------

def make_J0_quadrature(n_theta=256, dtype=jnp.float64):
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    theta = 0.5 * (nodes + 1.0) * np.pi
    w = 0.5 * np.pi * weights

    cth = jnp.asarray(np.cos(theta), dtype=dtype)  # precompute in numpy
    w   = jnp.asarray(w, dtype=dtype)
    inv_pi = jnp.asarray(1.0 / np.pi, dtype=dtype)

    @jax.jit
    def _jv_0(x):
        x = jnp.asarray(x, dtype=dtype)
        integrand = jnp.cos(x[..., None] * cth[None, :])
        return inv_pi * jnp.sum(integrand * w, axis=-1)

    return _jv_0

_jv_0 = make_J0_quadrature(n_theta=2048, dtype=jnp.float64)



def _jv_half(kappa, j):
    is_mhalf = jnp.isclose(kappa, -0.5)
    is_phalf = jnp.isclose(kappa,  0.5)
    is_zero = jnp.isclose(kappa,  0.0)

    def cos_branch(x):
        x_safe = jnp.where(x == 0.0, 1.0, x)
        val = jnp.sqrt(2.0 / (x_safe * jnp.pi)) * jnp.cos(x_safe)
        # J_{-1/2}(x) ~ sqrt(2/(pi x)) diverges at 0; keep it as +inf (or 0 if you prefer)
        return jnp.where(x == 0.0, jnp.inf, val)

    def sin_branch(x):
        x_safe = jnp.where(x == 0.0, 1.0, x)
        val = jnp.sqrt(2.0 / (x_safe * jnp.pi)) * jnp.sin(x_safe)
        # correct limit: J_{1/2}(0) = 0
        return jnp.where(x == 0.0, 0.0, val)
    
    def zero_branch(x):
        return _jv_0(x)

    def unsupported(x): return jnp.nan * x

    return jax.lax.cond(
        is_zero,
        zero_branch,
        lambda x_: jax.lax.cond(
            is_mhalf,
            cos_branch,
            lambda x__: jax.lax.cond(is_phalf, sin_branch, unsupported, x__),
            x_
        ),
        j
    )

def _jv_half_prime(kappa, j):
    is_mhalf = jnp.isclose(kappa, -0.5)
    is_phalf = jnp.isclose(kappa,  0.5)

    def deriv_mhalf(x):
        x_safe = jnp.where(x == 0.0, 1.0, x)
        sqrtfac = jnp.sqrt(2.0 / jnp.pi)
        val = sqrtfac * (-jnp.sin(x_safe) / jnp.sqrt(x_safe) - 0.5 * jnp.cos(x_safe) / (x_safe ** 1.5))
        return jnp.where(x == 0.0, -jnp.inf, val)  # it diverges

    def deriv_phalf(x):
        x_safe = jnp.where(x == 0.0, 1.0, x)
        sqrtfac = jnp.sqrt(2.0 / jnp.pi)
        val = sqrtfac * (jnp.cos(x_safe) / jnp.sqrt(x_safe) - 0.5 * jnp.sin(x_safe) / (x_safe ** 1.5))
        # this diverges like 1/sqrt(x); but in your sums it's multiplied by powers of j, so overall can be finite
        return jnp.where(x == 0.0, jnp.inf, val)

    def unsupported(x): return jnp.nan * x

    return jax.lax.cond(
        is_mhalf,
        deriv_mhalf,
        lambda x_: jax.lax.cond(is_phalf, deriv_phalf, unsupported, x_),
        j,
    )

# --- helper: wrap a *scalar* function to accept scalar or any-shaped array ---
def _wrap_scalar_or_batch(func_scalar):
    """
    If r is scalar -> returns scalar.
    If r is array (any shape) -> vmaps over the flattened array and reshapes back.
    """
    func_scalar = jax.jit(func_scalar)  # keep the scalar core fast

    @jax.jit
    def f(r):
        r = jnp.asarray(r)
        if r.ndim == 0:
            return func_scalar(r)  # scalar in -> scalar out
        else:
            flat = r.reshape(-1)
            out_flat = jax.vmap(func_scalar)(flat)
            return out_flat.reshape(r.shape)
    return f

# -------------------------- helpers: DE map for Hankel (nodes/weights) --------------------------

def _de_params(M):
    beta = 0.25
    alpha = beta / jnp.sqrt(1.0 + M * jnp.log(1.0 + M) / (4.0 * jnp.pi))
    return alpha, beta

def _de_phi(t, M, alpha, beta):
    expo = -2.0*t + alpha*jnp.expm1(-t) - beta*jnp.expm1(t)
    denom = -jnp.expm1(expo)  # = 1 - exp(expo)
    val = t / denom
    return jnp.nan_to_num(val, nan=0.0)

def _de_phi_prime(t, M, alpha, beta):
    expo = -2.0*t + alpha*jnp.expm1(-t) - beta*jnp.expm1(t)
    eE = jnp.exp(expo)
    g  = -jnp.expm1(expo)                     # = 1 - exp(expo)
    Ep = -2.0 - alpha*jnp.exp(-t) - beta*jnp.exp(t)
    val = (g + t*eE*Ep) / (g**2)
    return jnp.nan_to_num(val, nan=0.0)

def _build_hankel_rule(d, frac_order, h=0.05, N=100, M=jnp.pi):
    kappa = 0.5*d - 1.0
    c0 = 0.5*kappa - 0.25
    alpha_de, beta_de = _de_params(M)

    n = jnp.arange(-N, N+1, dtype=jnp.float64)
    t   = n * h
    t_n = n * h + c0 * h

    phi_t   = _de_phi(t,   M, alpha_de, beta_de)
    phi_p   = _de_phi_prime(t,   M, alpha_de, beta_de)
    phi_t_n = _de_phi(t_n, M, alpha_de, beta_de)
    phi_p_n = _de_phi_prime(t_n, M, alpha_de, beta_de)

    j  = M * phi_t_n
    wj = M * phi_p_n
    s  = M * phi_t
    ws = M * phi_p

    return {"j": j, "wj": wj, "kappa": kappa, "c0": c0,
            "h": h, "N": N, "M": M, "alpha_de": alpha_de, "beta_de": beta_de,
            "s": s, "ws": ws}

# -------------------------- kernel Fourier transforms --------------------------

def _Fphi_gaussian_factory(eps, d):
    # φ(r)=exp(-eps r^2) ⇒  Fφ(ω) = (2 eps)^(-d/2) * exp(-|ω|^2 / (4 eps))
    scale = (2.0*eps)**(-0.5*d)
    def F(omega):
        return scale * jnp.exp(-(omega*omega) / (4.0*eps))
    return F

def _Fphi_matern_factory(a, nu, d):
    # φ(r) ∝ (a r)^ν K_ν(a r) ⇒  Fφ(ω) ∝ a^{2ν} Γ(ν+d/2)/Γ(ν) (a^2+ω^2)^{-ν-d/2}
    from jax.scipy.special import gamma
    pref = (2.0**(0.5*d)) * (a**(2.0*nu)) * (gamma(nu + 0.5*d) / gamma(nu))
    def F(omega):
        return pref * (a*a + omega*omega) ** (-nu - 0.5*d)
    return F

# -------------------------- main class --------------------------

class FractionalLaplacianRBF:
    """
    Evaluate the fractional Laplacian of a radial basis function via DE–Hankel quadrature.

    Parameters
    ----------
    d : int
        Space dimension.
    frac_order : float
        Fractional order α (>0 typically).
    kernel : {'gaussian','matern'}
        RBF family (affects Fφ).
    kernel_params : dict
        For 'gaussian': {'eps': ...}
        For 'matern'  : {'a': ..., 'nu': ...}
    h, N, M : float, int, float
        DE rule step, half-window count, scaling.
    attach_custom_grad : bool
        If True and d in {1,3}, attach analytic JVP so jax.grad(eval) uses `derivative`.
    """

    def __init__(self, d, frac_order, kernel, kernel_params, *,
                 h=0.05, N=100, M=None):
        if not d in (1, 2, 3):
            raise NotImplementedError("Currently only d=1,2,3 are supported.")
        
        self.d = float(d)
        self.alpha = float(frac_order)
        self.kernel = kernel.lower()
        self.kernel_params = dict(kernel_params)
        if M is None:
            M = jnp.pi / h
        
        if int(self.d) == 1:
            self.Jk = lambda x: _jv_half(-0.5, x)   # closed form
        elif int(self.d) == 2:
            self.Jk = _jv_0                         # your quadrature J0
        elif int(self.d) == 3:
            self.Jk = lambda x: _jv_half(0.5, x)    # closed form
        else:
            raise NotImplementedError("Currently only d=1,2,3 are supported.")

        # quadrature precompute
        self.quad = _build_hankel_rule(self.d, self.alpha, h=h, N=N, M=jnp.array(M))

        # choose Fφ
        if self.kernel == 'gaussian':
            eps = float(self.kernel_params['eps'])
            self.Fphi = _Fphi_gaussian_factory(eps, self.d)
        elif self.kernel == 'matern':
            a  = float(self.kernel_params['a'])
            nu = float(self.kernel_params['nu'])
            self.Fphi = _Fphi_matern_factory(a, nu, self.d)
        else:
            raise ValueError("kernel must be 'gaussian' or 'matern'.")

        # build scalar evaluators
        self._f_scalar = self._make_scalar_eval()
        if int(self.d) in (1, 3):
            self._df_scalar = self._make_scalar_deriv()
        else:
            self._df_scalar = None  # derivative not implemented for d=2, will use finite difference
            
        self._attach_custom_jvp()

    # --------- core (scalar) formulas, reusing your implementations ----------

    def _fraclap_r_scalar(self, r):
        j = self.quad["j"]; wj = self.quad["wj"]; h = self.quad["h"]
        d = self.d; alpha = self.alpha; kappa = self.quad["kappa"]

        def body_for_rpos(rr):
            power = (0.5*d + alpha)
            Jk = self.Jk(j)
            Fvals = self.Fphi(j / rr)
            summand = (j**power) * Fvals * Jk * (wj / rr)
            return (h * rr**(-alpha + 1.0 - d)) * jnp.sum(summand)

        def body_r0(_):
            # r=0 branch per your comment (constant prefactor kept as 1 here)
            power = (0.5*d + alpha)
            s  = self.quad["s"]
            ws = self.quad["ws"]
            Fvals = self.Fphi(s)
            return h * jnp.sum((s**power) * Fvals * ws)

        return jax.lax.cond(r > 1e-11, body_for_rpos, body_r0, r)

    def _fraclap_r_prime_scalar(self, r):
        # only valid for d in {1,3}
        d = self.d; alpha = self.alpha
        assert int(d) in (1, 3)
        kappa = 0.5 * d - 1.0
        j = self.quad["j"];  wj = self.quad["wj"];  h = self.quad["h"]
        rr = r

        invj = 1.0 / jnp.where(j == 0.0, jnp.inf, j)
        Jk = _jv_half(kappa, j)
        Jkm1 = _jv_half_prime(kappa, j) + kappa * invj * Jk  # J_{κ-1}(j)

        pA = 0.5 * d + alpha + 1.0
        pB = 0.5 * d + alpha

        Fvals = self.Fphi(j / rr)
        S1 = jnp.sum((j**pA) * Fvals * Jkm1 * wj)
        S2 = jnp.sum((j**pB) * Fvals * Jk    * wj)

        term1 = (rr**(-alpha - d - 2.0)) * S1
        term2 = (1.0 + alpha) * (rr**(-alpha - d - 1.0)) * S2
        return h * (term1 - term2)

    # --------- wrappers built once, then vmapped/jitted above ----------

    def _make_scalar_eval(self):
        return jax.jit(lambda r: self._fraclap_r_scalar(r))

    def _make_scalar_deriv(self):
        return jax.jit(lambda r: self._fraclap_r_prime_scalar(r))

    # --------- optional: attach analytic JVP so grad(eval)=derivative ---------

    def _attach_custom_jvp(self):
        if self._df_scalar is None:
            self.eval = jax.jit(_wrap_scalar_or_batch(self._f_scalar))
            self.derivative = None
            return
        # Build a scalar function with custom_jvp attached and then vmap it.
        d_eval = self._df_scalar  # analytic derivative (scalar)

        @jax.custom_jvp
        def f_scalar_with_jvp(r):
            return self._f_scalar(r)

        @f_scalar_with_jvp.defjvp
        def _jvp(primals, tangents):
            (r,), (r_dot,) = primals, tangents
            y = self._f_scalar(r)
            dy_dr = d_eval(r)
            return y, dy_dr * r_dot

        # replace eval with the custom-JVP version
        self.eval = jax.jit(_wrap_scalar_or_batch(f_scalar_with_jvp))
        self.derivative = jax.jit(_wrap_scalar_or_batch(d_eval))

# -------------------------- example usage --------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate fracLapRBF_example_output.npz")
    parser.add_argument('--d', type=int, default=3, help='Space dimension (1 or 3)')
    parser.add_argument('--frac_order', type=float, default=1.0, help='Fractional order α')
    parser.add_argument('--kernel', type=str, default='gaussian', help="RBF kernel: 'gaussian' or 'matern'")
    parser.add_argument('--eps', type=float, default=1.0, help="Epsilon parameter for Gaussian kernel")
    parser.add_argument('--a', type=float, default=1.0, help="Parameter a for Matern kernel")
    parser.add_argument('--nu', type=float, default=1.5, help="Parameter nu for Matern kernel")
    parser.add_argument('--h', type=float, default=0.01, help='DE step size h')
    parser.add_argument('--N', type=int, default=1000, help='DE half-window N')
    parser.add_argument('--M', type=float, default=None, help='DE scaling M (default: pi/h)')
    args = parser.parse_args()

    fl = FractionalLaplacianRBF(
        d=args.d,
        frac_order=args.frac_order,
        kernel=args.kernel,
        kernel_params={'eps': args.eps, 'a': jnp.sqrt(2 * args.nu), 'nu': args.nu},
        h=args.h, N=args.N, M=args.M
    )
    r = jnp.arange(0.0025, 6.0, 0.0025)
    y_grid = fl.eval(r)
    # dy_grid = fl.derivative(r)
    # use finite difference to compute derivative
    dy_grid = (fl.eval(r + 1e-6) - fl.eval(r - 1e-6)) / (2e-6)

    # concatenate r= 0
    r = jnp.concatenate((jnp.array([0.0]), r))
    y_grid = jnp.concatenate((y_grid[0:1], y_grid))
    dy_grid = jnp.concatenate((dy_grid[0:1], dy_grid))
    # save this into a npz file
    if fl.kernel == 'gaussian':
        jnp.savez(f"fracLapRBF_d_{int(fl.d)}_frac_order_{int(fl.alpha*10)}_gaussian.npz", r=r, y=y_grid, dy=dy_grid)
    elif fl.kernel == 'matern':
        if jnp.isclose(args.nu, 1.5):
            jnp.savez(f"fracLapRBF_d_{int(fl.d)}_frac_order_{int(fl.alpha*10)}_matern32.npz", r=r, y=y_grid, dy=dy_grid)
        elif jnp.isclose(args.nu, 2.5):
            jnp.savez(f"fracLapRBF_d_{int(fl.d)}_frac_order_{int(fl.alpha*10)}_matern52.npz", r=r, y=y_grid, dy=dy_grid)
        else:
            raise NotImplementedError("Only nu=1.5 and nu=2.5 are implemented for fractional Matern kernel.")
    else:
        raise ValueError("kernel must be 'gaussian' or 'matern'.")

