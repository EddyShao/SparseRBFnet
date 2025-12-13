import jax
import jax.numpy as jnp

def make_interp1d_with_custom_deriv(xg, yg, dyg, *, fill_value=0.0):
    """
    1D linear interpolant on an increasing grid xg with custom derivative dyg.
    - Out-of-bounds values and gradients are 0.
    - Works with jax.grad and jax.jit.
    """
    xg = jnp.asarray(xg); yg = jnp.asarray(yg); dyg = jnp.asarray(dyg)
    n = xg.size
    x_min, x_max = xg[0], xg[-1]

    def locate_scalar(x):
        # find i so xg[i] <= x < xg[i+1]
        i = jnp.searchsorted(xg, x, side="right") - 1
        i = jnp.clip(i, 0, n - 2)
        dx = xg[i+1] - xg[i]
        t = (x - xg[i]) / dx
        return i, t, dx

    def in_bounds_scalar(x):
        return (x >= x_min) & (x <= x_max)

    def value_scalar(x):
        i, t, _ = locate_scalar(x)
        y = (1.0 - t) * yg[i] + t * yg[i + 1]
        return jnp.where(in_bounds_scalar(x), y, fill_value)

    def slope_scalar(x):
        i, t, _ = locate_scalar(x)
        s = (1.0 - t) * dyg[i] + t * dyg[i + 1]
        return jnp.where(in_bounds_scalar(x), s, 0.0)

    @jax.custom_jvp
    def f_scalar(x):
        return value_scalar(x)

    @f_scalar.defjvp
    def _jvp(primals, tangents):
        (x,), (xdot,) = primals, tangents
        y = value_scalar(x)
        dydx = slope_scalar(x)
        return y, dydx * xdot

    # single-argument API (scalar or 1D array)
    @jax.jit
    def f(x):
        x = jnp.asarray(x)
        if x.ndim == 0:
            return f_scalar(x)
        elif x.ndim == 1:
            return jax.vmap(f_scalar)(x)
        else:
            raise ValueError("x must be scalar or 1D.")
    return f


if __name__ == "__main__":
    data = jnp.load("src/frac/fracLapRBF_example_output.npz")
    r = data['r']
    y_grid = data['y']
    dy_grid = data['dy']

    f = make_interp1d_with_custom_deriv(r, y_grid, dy_grid)

    xq = jnp.array(0.37)
    x_space = jnp.linspace(0.01, 60.0, 60000)
    print(f(xq))            # interpolated value
    print(jax.grad(f)(xq))  # interpolated derivative

    # Evaluate on a grid
    y_space = f(x_space)
    # dy_space = jax.grad(f)(x_space)
    dy_space = jax.vmap(jax.grad(f))(x_space)

    def analytic_fraclap_gaussian(r, d=1, eps=1.0):
        return -(4.0 * (eps**2) * (r**2) - 2.0 * d * eps) * jnp.exp(-eps * r**2)
    

    

    y_analytic = analytic_fraclap_gaussian(x_space)
    dy_analytic = jax.vmap(jax.grad(lambda r: analytic_fraclap_gaussian(r)))(x_space)




    import matplotlib.pyplot as plt
    # plt.plot(x_space, y_space, label="y")
    plt.plot(x_space, dy_space, label="dy/dx")
    # plt.plot(x_space, y_analytic, '--', label="y analytic")
    plt.plot(x_space, dy_analytic, '--', label="dy/dx analytic")

    plt.legend()
    plt.show()



