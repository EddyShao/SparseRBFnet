
# import numpy as np
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from ..utils import computeProx, Phi, compute_rhs, compute_y, compute_errors
import jax 
# from memory_profiler import profile

# jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision    

Prox = lambda v: computeProx(v, mu=1)


# @profile
def solve(p, y_ref, alg_opts):
    """
    Solve the Total Variation problem
    Gauss-Newton (active point) method for TV-regularized

    Parameters
    ----------
    p : problem class object
    y_ref : torch.Tensor
        Reference points
    alpha : float
        Regularization parameter
    phi : function
        Function handle for the forward operator
    alg_opts : dict
        Dictionary containing the algorithm options

    """


    # redefine for readability
    d = p.d
    dim = p.dim 
    obj = p.obj 

    Ndata = len(y_ref)

    max_step = alg_opts.get('max_step', 1000)
    TOL = alg_opts.get('TOL', 1e-5)
    insertion_coef = alg_opts.get('insertion_coef', 0.01)
    gamma = alg_opts.get('gamma', 1)
    alpha = alg_opts.get('alpha', 0.1)

    plot_final = alg_opts.get('plot_final', True)
    plot_every = alg_opts.get('plot_every', 0)
    print_every = alg_opts.get('print_every', 20)
    blocksize  = alg_opts.get('blocksize', 1000)

    Ntrial = alg_opts.get('Ntrial', 1000)
    T = alg_opts.get('T', 300)

   

    ck = jax.random.normal(jax.random.PRNGKey(p.seed), shape=p.u_zero['u'].shape) 

    xk, sk = p.sample_param(len(ck))
    yk, linear_results_int, linear_results_bnd = compute_rhs(p, xk, sk, ck)
    misfit = yk - y_ref
    j = obj.F(misfit) / alpha + jnp.sum(ck**2)
    errors = compute_errors(p, xk, sk, ck)

    alg_out = {
        'xk': [xk],
        'sk': [sk],
        'ck': [ck],
        'suppc': [jnp.ones_like(ck, dtype=bool)],
        'supps': [len(ck)],
        'js': [j],
        'tics': [0],
        'L_2': [errors['L_2']],
        'L_inf': [errors['L_inf']]
    }

    start_time = time.time()
    shape_dK = lambda dK: dK.transpose(0, 2, 1).reshape(Ndata, -1) 

    

    # Define function equivalen
    theta_old = 1. # initial step size for line search

    print('### Start Iterations ###')
    
    pad_size = xk.shape[0]
    hessian_ck_2 = jnp.concatenate([2 * jnp.ones((pad_size,)), jnp.zeros((dim * pad_size))])
    
    for k in range(1, max_step  + 1):
        Grad_E = p.kernel.Grad_E_kappa_X_c_Xhat(xk, sk, ck, p.xhat_int)
        Grad_B = p.kernel.Grad_B_kappa_X_c_Xhat(xk, sk, ck, p.xhat_bnd)
        
        Dc_E_kappa, Dx_E_kappa, Ds_E_kappa = Grad_E['grad_c'], Grad_E['grad_X'], Grad_E['grad_S']
        Dc_B_kappa, Dx_B_kappa, Ds_B_kappa = Grad_B['grad_c'], Grad_B['grad_X'], Grad_B['grad_S']
        
        Gp_c = jnp.vstack([Dc_E_kappa, Dc_B_kappa])
        Gp_x = jnp.vstack([Dx_E_kappa, Dx_B_kappa])
        Gp_s = jnp.vstack([Ds_E_kappa, Ds_B_kappa])
        
        if Gp_s.ndim == 2:
            Gp_s = Gp_s[:, :, None]
        Gp_xs = jnp.dstack([Gp_x, Gp_s])
        
        # We optimize over qk, xk, and sk
        Gp = jnp.hstack([Gp_c, shape_dK(Gp_xs)])


        R1 = (1 / alpha) * (Gp.T @ obj.dF(misfit))
        grad_ck_2 = jnp.concatenate([2 * ck, jnp.zeros((dim * pad_size))])
        R2 = grad_ck_2.reshape(-1, 1)
        R = R1 + R2  

        II = obj.ddF_quad(misfit, Gp) # Approximate Hessian


        kpp = 0.1 * jnp.linalg.norm(obj.dF(misfit), 1) * jnp.reshape(
            jnp.sqrt(jnp.finfo(float).eps) + jnp.tile(jnp.abs(ck), (dim, 1)), -1
        )
        

        Icor_diag = jnp.concatenate([
            jnp.sqrt(jnp.finfo(float).eps) * jnp.ones((pad_size,)),
            kpp
        ])

        II = II.at[jnp.diag_indices(II.shape[0])].add(Icor_diag)
        HH = (1 / alpha) * II

        
        DR = HH.at[jnp.diag_indices(HH.shape[0])].add(hessian_ck_2)

        try:

            # Diagonal preconditioner
            diag_DR = jnp.diag(DR)
            eps = 1e-12
            sqrt_d = jnp.sqrt(jnp.abs(diag_DR) + eps)
            inv_sqrt_d = 1.0 / sqrt_d

            # Symmetric scaling in-place (conceptually): DR <- D^{-1/2} DR D^{-1/2}
            DR = (inv_sqrt_d[:, None] * DR) * inv_sqrt_d[None, :]
            # Scale RHS: R <- D^{-1/2} R
            R_scaled = inv_sqrt_d.reshape(-1, 1) * R

            # Solve preconditioned system
            dz = -jnp.linalg.solve(DR, R_scaled)

            # Map back: dz = D^{-1/2} dz_scaled
            dz = inv_sqrt_d.reshape(-1, 1) * dz
            dz = dz.flatten()

        except:
            print("WARNING: Singular matrix encountered.")
            alg_out["success"] = False
            break
        
        # check if dz is finite
        assert jnp.all(jnp.isfinite(dz)), "dz contains NaN or Inf values, check the problem setup."
        # # check if dz is zero in the lower trunk
        # # This is a sanity check, it should be always true, otherwise the support definition is incorrect
        # assert jnp.all(jnp.abs(dz[int(jnp.sum(suppGp)):]) < 1e-14), "dz contains non-zero values in the lower trunk, check the problem setup."

        jold, xold, sold, cold = j, xk.copy(), sk.copy(), ck.copy()
        pred = (R.T @ dz.reshape(-1, 1)) # estimate of the descent
        theta = min(theta_old * 2, 1 - 1e-14) 
        has_descent = False
        
        while not has_descent and theta > 1e-20:
            ck = cold + theta * dz[:pad_size]
            dxs = dz[pad_size:].reshape(dim, -1).T
            xk = xold + theta * dxs[:, :d]
            sk = sold + theta * dxs[:, d:]

            yk, linear_results_int, linear_results_bnd = compute_rhs(p, xk, sk, ck)
            misfit = yk - y_ref
            j = obj.F(misfit) / alpha + jnp.sum(ck**2)
           
            descent = j - jold 
            has_descent = descent <= (theta*pred + 1e-5) / 5

            if not has_descent:
                theta /= 1.5 # shrink theta

        theta_old = theta 


    
        errors = compute_errors(p, xk, sk, ck)

        # Print iteration info
        if k % print_every == 0:
            dz_norm = jnp.linalg.norm(dz, jnp.inf) if dz.size > 0 else 0.0
            print(f"Time: {time.time() - start_time:.2f}s CGNAP iter: {k}, j={j:.6f}, "
                f"desc={descent:.1e}, dz={dz_norm:.1e}, theta={theta:.2e}") 
            
            print("L_2 error: {L_2:.3e}, L_inf error: {L_inf:.3e}, (int: {L_inf_int:.3e}, bnd: {L_inf_bnd:.3e})".format(**errors))
        

        alg_out["xk"].append(xk)
        alg_out["sk"].append(sk)
        alg_out["ck"].append(ck)
        alg_out["suppc"].append(jnp.ones_like(ck, dtype=bool))
        alg_out["supps"].append(len(ck))
        alg_out["js"].append(j)
        alg_out["L_2"].append(errors['L_2'])
        alg_out["L_inf"].append(errors['L_inf'])
        alg_out["tics"].append(time.time() - start_time) 
        alg_out["success"] = True


    
        # Plot results
        if k % plot_every == 0:       
            p.plot_forward(xk, sk, ck, jnp.abs(ck) > 0)

        # # Stopping criterion
        # if jnp.abs(pred * theta) < (TOL / alpha):
        #     dz_norm = jnp.linalg.norm(dz, jnp.inf) if dz.size > 0 else 0.0  
        #     print(f"Time: {time.time() - start_time:.2f}s CGNAP iter: {k}, j={j:.6f}, supp=({jnp.sum(suppc)}), "
        #         f"desc={descent:.1e}, dz={dz_norm:.1e}, ")
            
        #     print("L_2 error: {L_2:.3e}, L_inf error: {L_inf:.3e}, (int: {L_inf_int:.3e}, bnd: {L_inf_bnd:.3e})".format(**errors))   
        #     print(f"Converged in {k} iterations")
        #     break

        #     # update xhat_int and xhat_bnd, resampling
        if 'grid' not in alg_opts.get('sampling', 'uniform'):
            p.xhat_int, p.xhat_bnd = p.sample_obs(p.Nobs_int, p.Nobs_bnd, method=alg_opts.get('sampling', 'uniform'))
            # recompute j due to resampling
            y_ref = p.f(p.xhat)
            y_ref = y_ref.at[-p.Nx_bnd:].set(p.ex_sol(p.xhat_bnd))

            yk, linear_results_int, linear_results_bnd = compute_rhs(p, xk, sk, ck)
            misfit = yk - y_ref
            j = obj.F(misfit) / alpha + jnp.sum(ck**2)
        

    
    if plot_final:
        p.plot_forward(xk, sk, ck, jnp.abs(ck) > 0)
    

    return alg_out


