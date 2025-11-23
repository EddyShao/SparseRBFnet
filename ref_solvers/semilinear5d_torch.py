import os
os.environ["DDE_BACKEND"] = "pytorch"  # use PyTorch backend

import deepxde as dde
import numpy as np
from deepxde.backend import backend as bkd  # PyTorch backend under the hood
import torch
# Use float64 like JAX x64
dde.config.set_default_float("float64")

# ======================
#   Problem settings
# ======================

d = 5  # dimension

# ----- Exact solution: NumPy version (for BC + postproc) -----
def u_exact_np(x):
    """
    x: numpy array (N, d)
    returns: (N, 1)
    """
    return np.sum(np.sin(2 * 0.5 * np.pi * x), axis=1, keepdims=True)

# ----- Exact solution: backend version (for PDE / forcing) -----
def u_exact_backend(x):
    """
    x: backend tensor (PyTorch tensor when backend is pytorch)
    returns: (N, 1) tensor
    """
    return torch.sum(torch.sin(2 * 0.5 * torch.pi * x), dim=1, keepdims=True)

def f_backend(x):
    """
    f(x) = (pi^2 / 4) * u(x) + u(x)^3
    x: backend tensor
    returns: (N, 1) backend tensor
    """
    u = u_exact_backend(x)
    return 2**2 * (torch.pi**2 / 4.0) * u + u**3


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
    x: (N, d) backend tensor
    y: (N, 1) backend tensor (network output)
    """
    # Laplacian: sum_i d^2 u / dx_i^2
    lap = 0.0
    for i in range(d):
        dy_xx_i = dde.grad.hessian(y, x, i=i, j=i)
        lap = lap + dy_xx_i

    u = y
    f_val = f_backend(x)
    return -lap + u**3 - f_val


# ======================
#    Boundary condition
# ======================

def boundary(x, on_boundary):
    return on_boundary

def u_exact_bc(x):
    """
    x: NumPy array (DeepXDE converts this to backend tensor internally)
    """
    return u_exact_np(x)

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
    train_distribution="uniform",
)


# ======================
#        Network
# ======================

layer_sizes = [d] + [64] * 2 + [1]  # 8 hidden layers, width 64
net = dde.nn.FNN(layer_sizes, "tanh", "Glorot uniform")


# ======================
#        Model + L-BFGS
# ======================

model = dde.Model(data, net)

# Configure L-BFGS (optional but recommended)
from deepxde.optimizers import config as opt_config

opt_config.set_LBFGS_options(
    maxiter=10000,
    maxcor=50,
    ftol=0.0,
    gtol=1e-8,
    maxfun=None,
    maxls=50,
)

# Compile with L-BFGS ONLY — no lr, no Adam
model.compile(
    "L-BFGS",
    loss_weights=[1.0, 1000.0],
)

# For L-BFGS: DO NOT pass epochs/iterations
losshistory, train_state = model.train()


# ======================
#     Post-processing
# ======================

# Build 20^5 grid in [-1,1]^5, including boundary (NumPy version)
n_test_axis = 20
xs_1d = np.linspace(-1.0, 1.0, n_test_axis)

mesh = np.stack(
    np.meshgrid(xs_1d, xs_1d, xs_1d, xs_1d, xs_1d, indexing="ij"),
    axis=-1,
)  # shape: (20, 20, 20, 20, 20, 5)

X_test_np = mesh.reshape(-1, d)  # (20^5, 5) = (3,200,000, 5)

u_pred = model.predict(X_test_np)
u_true = u_exact_np(X_test_np)

l2_rel = dde.metrics.l2_relative_error(u_true, u_pred)
print(f"Relative L2 error on 20^5 grid: {l2_rel:.2e}")