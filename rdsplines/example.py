import sys
sys.path.append('../')

from rdsplines import array_lib as np
from plt import plt
from rdspline_fit import rdspline_fit
from generate_samples import generate_samples, adjust_samples_to_bcs, adjust_samples_to_grid, accept_reject_using_function


## Define domain and sensitivity

s_lim = 5
t_lim = 20
domain = np.array([[0, t_lim], [-s_lim, s_lim]])
bcs = ["wrap", "wrap"]

eps = 0.01
max_measure = eps + 0.25
def measure_func(xys):
    return eps + 0.25*np.sin(1.5+xys[...,1]/(float(coeffs_shape[1])/(2*np.pi)))**2
measure_factor = 2.7


## Rdspline parameters

lam = 2.
coeffs_shape = (44, 44)


## Generate samples

nsamples = 5000
nsamples = int(measure_factor*nsamples)
samples = generate_samples(nsamples, domain)
sample_coords = adjust_samples_to_bcs(samples, domain, bcs, plot=True)
scaled_coords = adjust_samples_to_grid(sample_coords, domain, coeffs_shape, plot=True)
measured_coords = accept_reject_using_function(scaled_coords, measure_func, max_measure, plot=True)
#plt.scatter(*samples.T, alpha=0.1); plt.show()


## Fit rdsplines

out_coeffs, out_eval, rds = rdspline_fit(measured_coords, coeffs_shape, bcs, measure_func = measure_func, lam=lam, verbose=0)
pl=plt.imshow(out_eval, aspect="auto"); plt.colorbar(pl); plt.show()
