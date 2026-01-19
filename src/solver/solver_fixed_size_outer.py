
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

    # Initial guess
    u0 = alg_opts.get('u0', p.u_zero)
    uk = u0.copy()

    phi = Phi(gamma)
    yk, linear_results_int, linear_results_bnd = compute_rhs(p, uk['x'], uk['s'], uk['u'])
    norms_c = jnp.abs(uk['u'])

    # Compute initial loss, error, and etc.
    misfit = yk - y_ref
    j = obj.F(misfit)/alpha + jnp.sum(phi.phi(norms_c))
    errors = compute_errors(p, uk['x'], uk['s'], uk['u'])
    suppsize = jnp.count_nonzero(norms_c) # support size

    ck = uk['u'] # outer weights

    xk, sk = p.sample_param(len(ck))

    alg_out = {
        'xk': [xk],
        'sk': [sk],
        'ck': [ck],
        'suppc': [jnp.abs(ck) > 0],
        'supps': [suppsize],
        'js': [j],
        'tics': [0],
        'L_2': [errors['L_2']],
        'L_inf': [errors['L_inf']]
    }

    start_time = time.time()
    shape_dK = lambda dK: dK.transpose(0, 2, 1).reshape(Ndata, -1) 

    # generate a random array of +-1
    # qk = jax.random.choice(jax.random.PRNGKey(p.seed), jnp.array([-1, 1]), shape=(len(ck),))
    # sample qk based on gaussian distribution centered at 0 
    qk = jax.random.normal(jax.random.PRNGKey(p.seed), shape=ck.shape) 
    
    qk = jnp.sign(qk) + qk # perturbation around +-1
    

    ck = Prox(qk) # Update ck, no change in default
    suppc = jnp.ones_like(ck, dtype=bool) # support of the outer weights
    suppGp = jnp.tile(suppc, dim+1) # support of the gradiento of the objective function

    # Define function equivalents
    Dphima = lambda c: (phi.dphi(jnp.abs(c)) - 1) * jnp.sign(c) # gradient of modified phi
    DDphima = lambda c: phi.ddphi(jnp.abs(c)) # hessian of modified phi


    theta_old = 1. # initial step size for line search

    print('### Start Iterations ###')
    
    pad_size = xk.shape[0]
    
    for k in range(1, max_step  + 1):
        Dc_E_kappa = p.kernel.Grad_c_E_kappa_X_c_Xhat(xk, sk, ck, p.xhat_int)
        Dc_B_kappa = p.kernel.Grad_c_B_kappa_X_c_Xhat(xk, sk, ck, p.xhat_bnd)

        Gp_c = jnp.vstack([Dc_E_kappa, Dc_B_kappa])
        Gp = Gp_c

        Gp = Gp * suppc[None, :] # only keep the active points #### TODO: ADD ACTIVE POINTS SETTING ####

        compact_ind = jnp.argsort(~suppc) # index used to shift all the active points to the front

        if blocksize > 0: # if -1 that means no blocksize limit
            compact_ind = compact_ind[:blocksize]

        inv_compact_ind = jnp.argsort(compact_ind) # index used to shift all the active points back to the original order
      
        Gp_compact = Gp[:, compact_ind] # compacted gradient matrix 

        R1 = (1 / alpha) * (Gp_compact.T @ obj.dF(misfit)) 
        R2 = jnp.concatenate([
            ((Dphima(ck)+qk-ck) * suppc).reshape(-1, 1),
            jnp.zeros((pad_size * dim, 1))
        ]) 
        R = R1 + R2[compact_ind, :]

        # # assert the lower trunk of R is zero
        # # This is a sanity check, it should be always true, otherwise the support definition is incorrect
        # assert jnp.all(jnp.abs(R[int((dim+1)*jnp.sum(suppc)):, :]) < 1e-14), "The lower trunk of R is not zero, check the support definition."  s

        # SI = obj.ddF(misfit)
        # # II = Gp_compact.T @ SI @ Gp_compact # Approximate Hessian 
        II = obj.ddF_quad(misfit, Gp_compact) # Approximate Hessian



        Icor_diag = jnp.sqrt(jnp.finfo(float).eps) * suppc
        Icor_diag_compact = Icor_diag[compact_ind]
        # Icor = jnp.diag(Icor_diag_compact)

        # HH = (1 / alpha) * (II + Icor)
        II = II.at[jnp.diag_indices(II.shape[0])].add(Icor_diag_compact)
        HH = (1 / alpha) * II

        # DP_diag = jnp.concatenate([
        #     (jnp.abs(qk) >= 1),
        #     jnp.tile((jnp.abs(ck) > 0), (dim,))
        # ])
        DP_diag = (jnp.abs(qk) >= 1)
        DP_diag_compact = DP_diag[compact_ind]
        


        # DP = jnp.diag(DP_diag_compact)
        DDphi_diag = jnp.concatenate([DDphima(ck), jnp.zeros((dim * pad_size))]) * suppGp
        DDphi_diag_compact = DDphi_diag[compact_ind]
        # DDphi = jnp.diag(DDphi_diag_compact)s

        try:
            # # DR = HH @ DP + DDphi @ DP + (jnp.eye(HH.shape[0]) - DP)
            # DR = HH*DP_diag_compact 
            # DR = DR.at[jnp.diag_indices(DR.shape[0])].add(DP_diag_compact*DDphi_diag_compact + (1 - DP_diag_compact))
            # # print(f"Condition number of DR: {jnp.linalg.cond(DR):.2e}")
            # dz = - jnp.linalg.solve(DR, R)
            # dz = dz.flatten()

            DR = HH * DP_diag_compact
            DR = DR.at[jnp.diag_indices(DR.shape[0])].add(
                DP_diag_compact * DDphi_diag_compact + (1.0 - DP_diag_compact)
            )

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
            try:
                # DR = HH @ DP + (jnp.eye(pad_size*(1+dim)) - DP)
                DR = HH*DP_diag_compact 
                DR = DR.at[jnp.diag_indices(DR.shape[0])].add((1 - DP_diag_compact) + jnp.finfo(float).eps)
                dz = - jnp.linalg.solve(DR, R)
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

        jold, xold, sold, qold = j, xk.copy(), sk.copy(), qk.copy()
        pred = (R.T @ (DP_diag_compact * dz).reshape(-1, 1)) # estimate of the descen
        theta = min(theta_old * 2, 1 - 1e-14) 
        has_descent = False

        # dz = dz[inv_compact_ind] # reorder dz to the original order
        dz = jnp.zeros(suppc.shape[0]).at[compact_ind].set(dz) # reorder dz to the original order
        dz = dz * suppc # only keep the active points redundant, but safe
        while not has_descent and theta > 1e-20:
            # qk = qold + theta * dz[:pad_size]
            qk = qold + theta * dz
            # dxs = dz[pad_size:].reshape(dim, -1).T
            # xk = xold + theta * dxs[:, :d]
            # sk = sold + theta * dxs[:, d:]
            ck = Prox(qk)

            yk, linear_results_int, linear_results_bnd = compute_rhs(p, xk, sk, ck)
            misfit = yk - y_ref
            norms_c = jnp.abs(ck)
            j = obj.F(misfit) / alpha + jnp.sum(phi.phi(norms_c))
           
            descent = j - jold 
            has_descent = descent <= (theta*pred + 1e-5) / 5

            if not has_descent:
                theta /= 1.5 # shrink theta

        theta_old = theta 

        # if not has_descent:
        #     print("WARNING: Line search failed.")
        #     alg_out["success"] = False
        #     break

        # suppc_new = jnp.abs(qk) > 1

        # if jnp.sum(suppc_new) < jnp.sum(suppc):
        #     print(f" PRUNE: supp:{jnp.sum(suppc)}->{jnp.sum(suppc_new)}")
        #     suppc = suppc_new
        #     suppGp = jnp.tile(suppc, dim+1)
            

    
        errors = compute_errors(p, xk, sk, ck)

        # Print iteration info
        if k % print_every == 0:
            dz_norm = jnp.linalg.norm(dz, jnp.inf) if dz.size > 0 else 0.0
            print(f"Time: {time.time() - start_time:.2f}s CGNAP iter: {k}, j={j:.6f}, supp={jnp.sum(jnp.abs(qk) > 1)}), "
                f"desc={descent:.1e}, dz={dz_norm:.1e}, theta={theta:.2e}") 
            
            print("L_2 error: {L_2:.3e}, L_inf error: {L_inf:.3e}, (int: {L_inf_int:.3e}, bnd: {L_inf_bnd:.3e})".format(**errors))
        

        alg_out["xk"].append(xk)
        alg_out["sk"].append(sk)
        alg_out["ck"].append(ck)
        alg_out["js"].append(j)
        alg_out["L_2"].append(errors['L_2'])
        alg_out["L_inf"].append(errors['L_inf'])
        alg_out["supps"].append(jnp.sum(jnp.abs(qk) > 1))
        alg_out["suppc"].append(jnp.abs(qk) > 1)
        alg_out["tics"].append(time.time() - start_time) 
        alg_out["success"] = True


    
        # Plot results
        if k % plot_every == 0:       
            p.plot_forward(xk, sk, ck, jnp.abs(qk) > 1)

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
            j = obj.F(misfit)/alpha + jnp.sum(phi.phi(norms_c)) 
        

    
    if plot_final:
        p.plot_forward(xk, sk, ck, jnp.abs(qk) > 1)

    return alg_out


