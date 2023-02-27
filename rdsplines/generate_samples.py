from rdsplines import array_lib as np
import scipy.stats as css
import bs_utils as bu
from math import sqrt
from plt import plt

def generate_samples(nsamples, domain):
    t_lim = float(domain[0,1])
    s_lim = float(domain[1,1])
    nsamples = int(nsamples)
    nsamples_split = int(nsamples/2)
    
    ## Generate samples
    
    # Gaussians
    m1 = np.array([t_lim/2., -s_lim]).get(); cov1 = np.array([1., 4.]).get()
    m2 = np.array([9*t_lim/10, s_lim/2]).get(); cov2 = np.array([3., 3.]).get()
    g1 = css.multivariate_normal(mean = m1, cov = cov1)
    samples1 = g1.rvs(nsamples-nsamples_split)
    
    # Exponentials
    ex = css.laplace(loc=m2[0], scale=sqrt(cov2[0]/2))
    ey = css.laplace(loc=m2[1], scale=sqrt(cov2[1]/2))
    samples2 = np.array((ex.rvs(nsamples_split), ey.rvs(nsamples_split))).T

    # Uniforms
    ux = css.uniform(loc=1.5, scale=t_lim/5)
    uy = css.uniform(loc=-s_lim+1.5, scale=s_lim/2)
    samples3 = np.array((ux.rvs(nsamples_split), uy.rvs(nsamples_split))).T
    
    samples = np.vstack((samples1, samples2, samples3))
    
    return samples


def adjust_samples_to_bcs(samples, domain, bcs, plot=False):
    domain_range = np.diff(domain)
    domain_offset = domain[:,0]
        
    comparison_domain = np.array([np.array([-np.infty, np.infty]) if bc=="wrap" else r for r, bc in zip(domain, bcs)])
    sample_coords = samples[np.all(samples>comparison_domain[:,0], axis=1)*np.all(samples<comparison_domain[:,1], axis=1)]
    sample_coords = ((sample_coords - domain_offset) % domain_range.T) + domain_offset # offset do periodic and get offset back
    
    if plot:
        plt.scatter(*sample_coords.T, alpha=0.1); plt.show()
        
    return sample_coords


def adjust_samples_to_grid(sample_coords, domain, coeffs_shape, plot=False):
    domain_dim = len(domain)
    coeffs_domain = np.array(coeffs_shape)#-1
    
    new_domain = np.vstack((np.zeros(domain_dim), coeffs_domain)).T
    scaled_coords = bu.rescale(sample_coords, new_domain, old_domain=domain)
    
    if plot:
        plt.scatter(*scaled_coords.T, alpha=0.1); plt.xlim(0, coeffs_domain[0]); plt.xlim(0, coeffs_domain[1]); plt.show()

    return scaled_coords


def accept_reject_using_function(scaled_coords, measure_func, max_measure, plot=False):
    thin_prob = measure_func(scaled_coords)/max_measure
    points_to_keep = thin_prob > np.random.uniform(0, 1, len(scaled_coords))
    point_coords = scaled_coords[points_to_keep, :]
    
    if plot:
        plt.scatter(*point_coords.T, alpha=0.1); plt.show()
        
    return point_coords


def evaluate_periodic_function(coeffs_shape, pdf_eval, domain, Ax=1, step=1, measure_f=lambda x: 1.):
# step is how many points between the grid
# Ax is the number of grid points expanded
# assuming square new_domain

    domain_dim = len(domain)
    coeffs_domain = np.array(coeffs_shape)#-1
    
    new_domain = np.vstack((np.zeros(domain_dim), coeffs_domain)).T

    c = 0
    d = coeffs_shape[-1]

    a = np.mgrid[c-Ax:d+Ax:step, c-Ax:d+Ax:step]
    new_side = a.shape[-1]
    a = a.reshape(2,-1).T

    b = bu.rescale(a, domain, old_domain=new_domain)
    ##(-c+d+2*x)*(1/step) == new_size
    pos = b.reshape(new_side,-1,2)
    eval_pdf = pdf_eval(pos)#np.array(g1.pdf(pos.get()) + g2.pdf(pos.get())) #*measure_f(pos)
    

    hh = np.zeros(coeffs_shape)
    x = int(Ax/step)
    hh = eval_pdf[x:-x,x:-x]
    hh[:x,...]+=eval_pdf[-x:,x:-x]
    hh[-x:,...]+=eval_pdf[:x,x:-x]
    hh[...,:x]+=eval_pdf[x:-x, -x:]
    hh[...,-x:]+=eval_pdf[x:-x, :x]
    #hh = hh.T[::-1,:]
    hh = hh*measure_f(np.mgrid[c:d:step, c:d:step][::-1].T)

    pl = plt.imshow(hh); plt.colorbar(pl); plt.show()

    return np.array(hh)