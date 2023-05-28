import sys
from rdsplines import array_lib as np
import bs_utils as bu
from descent_algorithms import APGD
from bs_Hessian import bs_Hessian
from HessianSchattenNorm import LpSchattenNorm, HessianSchattenNorm

def rdspline_fit(scaled_coords, coeffs_shape, bcs, measure_func = lambda xys: np.ones_like(xys[...,0]),
                    deg=1, integration_scale=1, init_coeffs=None, prox_iter=50, lam=10., apgd_iter=50, verbose=0):
    
    # Initialize
    if init_coeffs is None: init_coeffs = -np.ones(coeffs_shape)
    step = 1/integration_scale
    measure = measure_func(np.mgrid[0:coeffs_shape[1]:step, 0:coeffs_shape[0]:step].T).T
    measure_s = measure_func(np.mgrid[0:coeffs_shape[1], 0:coeffs_shape[0]].T).T

    # Create likelihood
    nlsl = bu.neg_rdspline_likelihood(scaled_coords, coeffs_shape, deg, integration_scale, bcs,
                                    measure=measure, measure_s=measure_s,
                                    updater={"approx":False, "u_multi":1., "use_circles":True})
    
    # Initialize Lip
    nlsl.jacobianT(init_coeffs) # load the integrals
    nlsl.update_lip(init_coeffs)#, approx=False, use_circles=True)

    # Create Hessian Schatten reg
    deriv_order = 2 # Hessian
    bs_filters = [bu.bs_deriv_filter(i) for i in range(deriv_order+1)]
    bs_filters[0] = bu.bs_to_filter(deg, int_grid=True)
    bs_filters[1]= np.array([1., 0 ,-1.])/2.

    Hessian = bs_Hessian(coeffs_shape, bs_filters, bcs)
    Schatt_Norm = LpSchattenNorm(Hessian.hess_shape, p=1, hermitian=True, flat=True, flat_input=False)
    Hessian_Schatt_Norm = HessianSchattenNorm(Hessian, Schatt_Norm, prox_iter, lam)

    # Accelerated proxima-gradient descent
    out_coeffs = APGD(nlsl, Hessian_Schatt_Norm, niter=apgd_iter, verbose=verbose, update_ilip=True, restart_method="RESTART")
    
    out_eval = bu.eval_rdspline_on_subgrid(out_coeffs, deg, 1, bcs)/nlsl.density_integral
    
    return out_coeffs, out_eval, nlsl
