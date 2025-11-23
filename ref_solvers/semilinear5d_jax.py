import os
os.environ["DDE_BACKEND"] = "jax"
os.environ["JAX_ENABLE_X64"] = "True"   # ensure env-level control

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import deepxde as dde
import jax.numpy as jnp
from deepxde.backend import backend as bkd


# ======================
#   Problem settings
# ======================

d = 5  # dimension; change as you like

def u_exact_backend(x):
    """
    u(x) = sum_{i=1}^d sin(pi/2 * x_i)
    x: (N, d) JAX array
    returns: (N, 1) JAX array
    """
    return jnp.sum(jnp.sin(2*0.5 * jnp.pi * x), axis=1, keepdims=True)


def f_backend(x):
    """
    f(x) = (pi^2 / 4) * u(x) + u(x)^3
    consistent with the semilinear PDE -Δu + u^3 = f
    x: (N, d) JAX array
    returns: (N, 1) JAX array
    """
    u = u_exact_backend(x)
    return 2**2 * (jnp.pi**2 / 4.0) * u + u**3

# For postprocessing (numpy version)
def u_exact_np(x):
    """
    x: numpy array (N, d)
    returns: (N, 1)
    """
    return np.sum(np.sin(2*0.5 * np.pi * x), axis=1, keepdims=True)


# ======================
#        Geometry
# ======================

xmin = [-1.0] * d
xmax = [1.0] * d
geom = dde.geometry.Hypercube(xmin, xmax)


# ======================
#          PDE
# ======================



def pde(x, y):
    """
    Residual: -Δu + u^3 = f(x)
    x: (N, d)
    y: could be tensor or a tuple (tensor, aux), depending on backend.
    """

    # Keep the original object for Hessian calls
    y_for_hessian = y

    # Extract the actual network output for algebraic operations
    if isinstance(y, (tuple, list)):
        u = y[0]   # actual NN output of shape (N, 1)
    else:
        u = y

    # Laplacian: sum_i d^2 u / dx_i^2
    lap = 0.0
    for i in range(d):
        # JAX backend: hessian returns (value, aux)
        dy_xx_i, _ = dde.grad.hessian(y_for_hessian, x, i=i, j=i)
        lap = lap + dy_xx_i

    f_val = f_backend(x)  # should be (N,1)
    return -lap + u**3 - f_val
# ======================
#    Boundary condition
# ======================

def boundary(x, on_boundary):
    return on_boundary


def u_exact_bc(x):
    return u_exact_backend(x)


bc = dde.icbc.DirichletBC(geom, u_exact_bc, boundary)


# ======================
#         Data
# ======================


data = dde.data.PDE(
    geom,
    pde,
    bc,
    num_domain=7775,
    num_boundary=1440,
    num_test=1000,
    train_distribution="uniform",      # <-- THIS is equispaced grid
)


# ======================
#        Network
# ======================

# DeepXDE will pick the JAX FNN when backend is set to JAX
layer_sizes = [d] + [64] * 8 + [1]  # 4 hidden layers, width 64
net = dde.nn.FNN(layer_sizes, "tanh", "Glorot uniform")


# ======================
#        Model
# ======================

model = dde.Model(data, net)

model.compile(
    "adam",
    lr=5e-4,
    loss_weights=[1.0, 1000.0],   # ✅ THIS is where it belongs
)


# use LBFGS optimizer from JAX optax
# model.compile(
#     "L-BFGS",
# )

losshistory, train_state = model.train(epochs=30000)

# Optionally fine-tune with (JAX) L-BFGS if enabled for the backend:
# model.compile("L-BFGS")
# losshistory, train_state = model.train()


# ======================
#     Post-processing
# ======================

# ======================
#     Post-processing
# ======================

# Build 20^5 grid in [-1,1]^5, including boundary
n_test_axis = 20
xs_1d = jnp.linspace(-1.0, 1.0, n_test_axis)

# meshgrid with indexing='ij' to get tensor-product grid
mesh = jnp.stack(
    jnp.meshgrid(xs_1d, xs_1d, xs_1d, xs_1d, xs_1d, indexing="ij"),
    axis=-1,
)  # shape: (20, 20, 20, 20, 20, 5)

X_test = mesh.reshape(-1, d)  # (20^5, 5) = (3,200,000, 5)

# DeepXDE's predict expects numpy array
X_test_np = np.array(X_test)

u_pred = model.predict(X_test_np)
u_true = u_exact_np(X_test_np)

l2_rel = dde.metrics.l2_relative_error(u_true, u_pred)
print("Relative L2 error on 20^5 grid:", l2_rel)