import os
os.environ["DDE_BACKEND"] = "pytorch"  # use PyTorch backend

import deepxde as dde
import numpy as np
from deepxde.backend import backend as bkd  # PyTorch backend under the hood
import torch

# Use float32
dde.config.set_default_float("float32")

# ======================
#   Problem settings
# ======================

d = 4  # dimension

# ----- Exact solution: NumPy version (for BC + postproc) -----
def u_exact_np(x):
    """
    x: numpy array (N, d)
    returns: (N, 1)
    u_exact(x) = prod_i (1 - x_i^2)^2
    """
    phi = 1.0 - x**2  # (N, d)
    return np.prod(phi**2, axis=1, keepdims=True)


# ----- Exact solution: backend version (for PDE / forcing) -----
def u_exact_backend(x):
    """
    x: torch.Tensor (N, d)
    returns: (N, 1)
    """
    phi = 1.0 - x**2
    return torch.prod(phi**2, dim=1, keepdim=True)


def laplacian_torch(u, x):
    """
    Compute Laplacian of scalar field u(x) using torch.autograd.
    u: (N, 1) tensor, depends on x
    x: (N, d) tensor with requires_grad=True
    returns: (N, 1) tensor
    """
    # ∇u: (N, d)
    grads = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    lap = 0.0
    for i in range(d):
        grad_i = grads[:, i:i+1]  # (N, 1)
        grad2_i = torch.autograd.grad(
            grad_i,
            x,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=True,
            retain_graph=True,
        )[0][:, i:i+1]  # (N, 1)
        lap = lap + grad2_i
    return lap


def f_backend(x):
    """
    f(x) = Δ^2 u_exact(x)
    x: torch.Tensor (N, d), requires_grad=True (DeepXDE ensures this)
    returns: (N, 1) tensor
    """
    # IMPORTANT: we must keep graph for Laplacian, but f does not depend
    # on NN parameters, only on x. That's fine: gradients wrt NN params
    # will not flow through u_exact.
    u_ex = u_exact_backend(x)                # (N, 1)
    lap_u = laplacian_torch(u_ex, x)         # Δu_exact
    bilap_u = laplacian_torch(lap_u, x)      # Δ(Δu_exact)
    return bilap_u


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
    Residual: Δ^2 u - f(x) = 0
    x: (N, d) torch tensor
    y: (N, 1) torch tensor (network output)
    """
    # First Laplacian of NN output
    lap_u = laplacian_torch(y, x)          # (N, 1)
    # Second Laplacian
    bilap_u = laplacian_torch(lap_u, x)    # (N, 1)

    f_val = f_backend(x)                   # (N, 1)
    return bilap_u - f_val                 # residual Δ^2 u - f


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


def navier_lap_bc(x, y, _):
    """
    Operator BC for Navier boundary condition:
        Δu(x) = Δu_exact(x) on ∂Ω.

    x: (N, d) torch tensor with requires_grad=True
    y: (N, 1) torch tensor = NN(x)
    """
    # Laplacian of NN output
    lap_u = laplacian_torch(y, x)          # (N, 1)

    # Laplacian of exact solution
    u_ex = u_exact_backend(x)             # (N, 1)
    lap_u_ex = laplacian_torch(u_ex, x)   # (N, 1)

    # Residual to be forced to 0 on the boundary
    return lap_u - lap_u_ex


# For now: enforce only u = u_exact on boundary (Dirichlet part).
# With the chosen bubble, this is u = 0 AND automatically ∂_n u = 0,
# so it is effectively clamped for this manufactured solution.
# bc_u = dde.icbc.DirichletBC(geom, u_exact_bc, boundary)
bc_u = dde.icbc.DirichletBC(geom, u_exact_bc, boundary)
bc_lap = dde.icbc.OperatorBC(geom, navier_lap_bc, boundary)

bcs = [bc_u, bc_lap]

# If you want to explicitly enforce ∂_n u = 0 (full clamped),
# you can add an OperatorBC like this (sketch):
#
# def normal_derivative(x, y, _):
#     # y: (N, 1), x: (N, d)
#     grads = []
#     for i in range(d):
#         du_dxi = dde.grad.jacobian(y, x, i=i)
#         grads.append(du_dxi)
#     grad_u = torch.cat(grads, dim=1)  # (N, d)
#     # Approximate outward normal as radial direction:
#     n = x / (torch.norm(x, dim=1, keepdim=True) + 1e-12)  # (N, d)
#     return torch.sum(grad_u * n, dim=1, keepdim=True)     # (N, 1), want = 0
#
# bc_slope = dde.icbc.OperatorBC(geom, normal_derivative, boundary)
#
# and then use [bc_u, bc_slope] below.


# ======================
#         Data
# ======================

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=1295,
    num_boundary=2799,
    num_test=1000,
    train_distribution="uniform",
)


# ======================
#        Network
# ======================

layer_sizes = [d] + [64] * 2 + [1]  # 2 hidden layers, width 64
net = dde.nn.FNN(layer_sizes, "tanh", "Glorot uniform")


# ======================
#        Model + L-BFGS
# ======================

model = dde.Model(data, net)

from deepxde.optimizers import config as opt_config

# opt_config.set_LBFGS_options(
#     maxiter=,
#     maxcor=50,
#     ftol=0.0,
#     gtol=1e-8,
#     maxfun=None,
#     maxls=50,
# )

# model.compile(
#     "L-BFGS",
#     loss_weights=[1.0],  # only one PDE residual here
# )

model.compile(
    "adam",
    lr=1e-3,                # start with 1e-3; can reduce to 1e-4 later
    loss_weights=[1.0, 1000., 1000.],   # weights for [PDE, BC u, BC lap
)

# Adam requires iterations / epochs
losshistory, train_state = model.train(iterations=10000)

# losshistory, train_state = model.train()


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

X_test_np = mesh.reshape(-1, d)  # (3,200,000, 5)

u_pred = model.predict(X_test_np)
u_true = u_exact_np(X_test_np)

l2_rel = dde.metrics.l2_relative_error(u_true, u_pred)
print(f"Relative L2 error on 20^5 grid: {l2_rel:.2e}")