import math
from rdsplines import array_lib as np
from rdsplines import scipy_lib_linalg
from rdsplines import scipy_lib_ndimage as nd
from rdsplines import scipy_lib_sg as sg
from plt import plt


def rescale(xys, new_domain, old_domain=None):

    a, b = new_domain[:,0], new_domain[:,1]

    if old_domain is None:
        maxi = np.max(xys, axis=0)
        mini = np.min(xys, axis=0)
    else:
        mini, maxi = old_domain[:,0], old_domain[:,1]

    return a + (xys-mini)*(b-a)/(maxi-mini)


def bs(eval, deg, positive=False):
    if positive:
        ax = eval
    else:
        ax = np.abs(eval)

    #h_sup = 0.5*(1+deg)
    if deg==0:
        return (ax<0.5)
    elif deg==1:
        return (ax<1.)*(1. - ax)
    elif deg==3:
        return (ax<2.)*(\
                         (ax<1)*(2./3. - ax**2 + 0.5*ax**3) \
                        +(ax>=1)*(1./6.)*((2. - ax)**3) \
                       )

    # Callables in piecewise not supported in cupy
    # elif deg==3:
    #     return np.piecewise(ax, \
    #                          [ax<1, ax>=1, ax>=2], \
    #                          [lambda x: 2./3. - x**2 + 0.5*x**3, \
    #                           lambda x: (1./6.)*((2. - x)**3), \
    #                           0])


def bs_to_filter(deg, scale=1, int_grid=True): # methods assume grid is regular so mayb no point
    half_domain = math.ceil((deg+1)/2)
    half_domain *= int(scale)
    if int_grid:
        half_domain -= 1 # edges are zero
    return bs(np.arange(-half_domain, half_domain+1, dtype=float)/scale, deg)


def bs_deriv_filter(deriv):
    pascal = [1.]
    for i in range(deriv):
        pascal.append(-1*pascal[i]*(deriv-i)/(i+1))
    return np.array(pascal)


def upsample_coeffs(coeffs, scale=1, bcs=None):
    up_size = (np.array(coeffs.shape)-1)*scale+1
    up_size = tuple(int(s) for s in up_size) # cupy compatibility
    up_coeffs = np.zeros(up_size) #autocast to tuple?
    #up_coeffs[::scale, ::scale] = coeffs
    sl = (np.s_[::scale],)*coeffs.ndim
    up_coeffs[sl] = coeffs
    return up_coeffs if bcs is None else np.pad(up_coeffs, tuple([(0,(scale-1)*(bc=="wrap")) for bc in bcs]))
    #np.pad(up_coeffs, tuple([(scale-1)*(bc=="wrap") for bc in bcs]))


def downsample_coeffs(coeffs, scale):
    #return coeffs[::scale, ::scale].copy()
    sl = (np.s_[::scale],)*coeffs.ndim
    return coeffs[sl].copy()


def convolve_coeffs(coeffs, bs_filters, bcs): # successive separable convolutions
    bs_conv = coeffs.copy()
    for i, bc in enumerate(bcs):
        bs_conv = nd.convolve1d(bs_conv, bs_filters[i], mode=bc, axis=i) #oaconvolve #look for convolves that changes fft or mult according to speed
    return bs_conv


def eval_bspline_on_grid(coeffs, deg, bcs, scale=1): # scale included just for some tricks
    bs_filters = np.array(coeffs.ndim*[bs_to_filter(deg, scale=scale)])
    return convolve_coeffs(coeffs, bs_filters, bcs)


def eval_bspline_on_subgrid(coeffs, deg, scale, bcs):
    bs_filters = np.array(coeffs.ndim*[bs_to_filter(deg, scale)])
    up_coeffs = upsample_coeffs(coeffs, scale, bcs=bcs)
    return convolve_coeffs(up_coeffs, bs_filters, bcs)


def bcs_to_value(bcs, periods=None, domain_shape=np.infty):
    if all(bc=="wrap" for bc in bcs) and (periods is None) and (domain_shape is not np.infty): return np.array(domain_shape)
    if hasattr(periods, "__iter__"): periods = list(periods)[::-1] # flip because pop later
    ct = 2*np.max(np.array(domain_shape)) # so that % does not affect the non-periodic dimensions; infinite=sys.maxsize-1 #np.infty (casts to float)
    return np.array([bc_to_value(bc, periods, ct) for bc in bcs])


def bc_to_value(bc, periods=1, ct=0):
    if bc == "wrap":
        return periods.pop() if hasattr(periods, "__iter__") else periods
    elif bc == "constant":
        return ct
    else:
        raise ValueError("This kind of boundary condition is not supported.") #ValueError or RuntimeError


def adjoint_conv_bcs(bcs):
    return [adjoint_conv_bc(bc) for bc in bcs]


def adjoint_conv_bc(bc):
    #if bc is not "wrap":
    #    bc = "constant"
    return bc


def spline_integral(coeffs, deg, scale, bcs, measure=1., out=False):
    values_on_grid = measure*eval_bspline_on_subgrid(coeffs, deg, scale, bcs)
    sum_of_values = np.sum(values_on_grid)/(scale**coeffs.ndim)
    return (sum_of_values, values_on_grid) if out else sum_of_values


def single_bspline_integral(support_dim, deg, power=1, scale=80):
    # compute for 1D bspline^power then power of support_dim because they are separable
    one_coeff = np.zeros(5); one_coeff[2]=1.
    _, v = spline_integral(one_coeff, deg, scale, ["wrap"], out=True)
    return (np.sum(v**power)/(scale**one_coeff.ndim))**support_dim


def eval_rdspline_on_subgrid(coeffs, deg, scale, bcs, measure=1.):
    return measure * np.exp(
                    eval_bspline_on_subgrid(coeffs, deg, scale, bcs)
                    )


def rdspline_integral(coeffs, deg, scale, bcs, measure=1., out=False):
    values_on_grid = eval_rdspline_on_subgrid(coeffs, deg, scale, bcs, measure=measure)
    sum_of_values = np.sum(values_on_grid)/(scale**coeffs.ndim)
    return (sum_of_values, values_on_grid) if out else sum_of_values


def plot_rdspline_integral_convergence(coeffs, deg, bcs, scales=4):
    hold = []
    for s in range(scales):
        hold.append(rdspline_integral(coeffs, deg, 2**s, bcs))
    plt.plot(hold); plt.show()
    return 0


def rdspline_betaintegral_vector(n_samples, coeffs, deg, scale, bcs, measure=1):
    density_integral, values_on_grid = rdspline_integral(coeffs, deg, scale, bcs, measure=measure, out=True)
    beta_integral = eval_bspline_on_grid(values_on_grid, deg, bcs, scale=scale)/(scale**coeffs.ndim) # mimicking convolution and then discaring wiht downsample
    return -n_samples*downsample_coeffs(beta_integral, scale)/density_integral


def loglikelihood(n_samples, coeffs, deg, scale, bcs, measure=1.):
    return  -n_samples*np.log(rdspline_integral(coeffs, deg, scale, bcs, measure=measure, out=False))


def overlapping_bs_filters_1d(deg, scale=1): #beta*beta filter
    supp = deg+1
    npoints = scale*supp+1-2
    #origin = int((npoints-1)/2)
    bf = bs_to_filter(deg, scale=scale)
    assert(npoints==len(bf))
    padin = npoints-1
    shifting_filter = scipy_lib_linalg.circulant(np.pad(bf, (0, padin))).T[:npoints,:npoints]
    
    return shifting_filter * bf#, origin


class rdspline_likelihood:
    def __init__(self, sample_coords, coeffs_shape, deg, scale, bcs, periods=None, measure=1., measure_s=1., updater=None):
        # scale is for the integral
        self.sample_coords = sample_coords; self.coeffs_shape = coeffs_shape;
        self.deg = deg; self.scale = scale; self.bcs = bcs; self.measure = measure;
        self.params = (self.deg, self.scale, self.bcs, self.measure)
        self.measure_s = measure_s if measure_s is not None else measure
        self.approx = False if updater is None else updater.pop("approx")
        self.u_multi = 1. if updater is None else updater.pop("u_multi")
        self.u_kwargs = updater

        self.nsamples = len(sample_coords)
        #self.ndim = sample_coords.ndim; assert self.ndim == len(coeffs_shape), "samples and coeffs do not match"
        self.ndim = len(coeffs_shape)
        self.support = self.deg + 1
        self.coeffs_size = np.prod(np.array(self.coeffs_shape))
        self.dim = [self.coeffs_shape, 1]

        self.density_integral = None
        self.beta_integral = None

        from rdsplines.sparse_sampler import bs_SparseSampler
        basis = lambda args: bs(args, self.deg)
        #bcs_behaviour = #bcs_to_value(bcs, 102, 2*np.max(np.array(coeffs_shape))) #############
        #self.bcs_behaviour = bcs_to_value(bcs, periods=periods, domain_shape=coeffs_shape)
        #self.spline_sampler = bs_SparseSampler(self.coeffs_shape, self.sample_coords, None, basis, self.support, self.bcs_behaviour)
        self.spline_sampler = bs_SparseSampler(self.coeffs_shape, self.sample_coords, None, basis, self.support, self.bcs)
        self.evald_bspline_sum = self.spline_sampler.sum_samples()

        self.betabeta_integral = single_bspline_integral(self.ndim, self.deg, power=2, scale=80) # for hessian norm approx.

    def __call__(self, coeffs, recompute=True):
        if recompute: self.density_integral = rdspline_integral(coeffs, *self.params, out=False)
        return (np.sum(self.spline_sampler(coeffs))
                - self.nsamples*np.log(self.density_integral)) # need to compute gradient first
    
    def gradient(self, coeffs):
        # coefficients are input
        # output is the gradient at the coeffs

        self.density_integral, self.values_on_grid = rdspline_integral(coeffs, *self.params, out=True)
        self.beta_integral = eval_bspline_on_grid(self.values_on_grid, self.deg, self.bcs, scale=self.scale)/(self.scale**coeffs.ndim) #trick to convolve
        self.beta_integral = downsample_coeffs(self.beta_integral, self.scale)/self.density_integral

        return (self.evald_bspline_sum 
                - self.nsamples*self.beta_integral)

    jacobianT = gradient

    def hessian_norm_approx(self, coeffs, max_average=False):
        # recompute self.beta_integral and values on grid?
        outer = np.sum(self.beta_integral*self.beta_integral)
        interaction_size = (self.deg*2+1)**self.ndim #2*(self.support-1) #can probably bound by self.deg*2-1
        if max_average:
            maxi = np.mean(nd.maximum_filter(self.values_on_grid, size=(self.deg*2+1), mode=self.bcs, origin=(0,0)))
        else:
            maxi = np.max(self.values_on_grid)
        other = np.sqrt(interaction_size * self.coeffs_size) * self.betabeta_integral * maxi / self.density_integral
        return self.nsamples*(outer + other)

    def hessian(self, coeffs):
        # recompute self.beta_integral?
        outer = np.sum(self.beta_integral*coeffs) * self.beta_integral
        return self.nsamples*(outer - other)
    
    def hessian_norm(self, coeffs, scale=1, use_circles=1):
        # recompute self.beta_integral?
        outer = np.sum(self.beta_integral*self.beta_integral)

        if scale is None: scale = self.scale
        hessian_terms = self.rdspline_betabeta_integral_tensor(coeffs, scale=scale)
        if use_circles:
            factor = 1. #float(use_circles)
            off_diag_radii = factor*np.sum(np.abs(hessian_terms[1:]), axis=0)
            other_max = np.max(hessian_terms[0]+off_diag_radii) # at 0 there are the diagonals
            other_min = np.min(hessian_terms[0]-off_diag_radii) 
            if use_circles>1:
                return self.nsamples*(np.array([np.array(0), outer]) + np.array([other_min, other_max])) # arraying 0 for cupy compatibility
            else:
                return float(self.nsamples*(outer + other_max))

        else:
            other = np.sqrt(np.sum(hessian_terms**2)) # fill in
            return self.nsamples*(outer + other)

    def rdspline_betabeta_integral_tensor(self, coeffs, scale=None, origin=0):
        if scale is None: scale = self.scale
        self.density_integral, self.values_on_grid = rdspline_integral(coeffs, self.deg, scale, self.bcs, self.measure_s, out=True)
        bbs_filters = overlapping_bs_filters_1d(self.deg, scale)
        #origin = int(len(bbs_filters[0]-1)/2)
        hessian_elements = []
        
        if self.ndim==2:
            for filter_i in bbs_filters:
                first_filtering = nd.convolve1d(self.values_on_grid, filter_i, mode=self.bcs[0], axis=0, origin=origin)
                for filter_j in bbs_filters:
                    second_filtering = nd.convolve1d(first_filtering, filter_j, mode=self.bcs[1], axis=1, origin=origin)
                    second_filtering = downsample_coeffs(second_filtering, scale)/(self.density_integral * (scale**coeffs.ndim))
                    hessian_elements.append(second_filtering)

        elif self.ndim==1:
            for filter_i in bbs_filters:
                second_filtering = nd.convolve1d(self.values_on_grid, filter_i, mode=self.bcs[0], axis=0, origin=origin)
                second_filtering = downsample_coeffs(second_filtering, scale)/(self.density_integral * (scale**coeffs.ndim))
                hessian_elements.append(second_filtering)
                
        elif self.ndim==3:
            for filter_i in bbs_filters:
                first_filtering = nd.convolve1d(self.values_on_grid, filter_i, mode=self.bcs[0], axis=0, origin=origin)
                for filter_j in bbs_filters:
                    second_filtering = nd.convolve1d(first_filtering, filter_j, mode=self.bcs[1], axis=1, origin=origin)
                    for filter_k in bbs_filters:
                        third_filtering = nd.convolve1d(second_filtering, filter_k, mode=self.bcs[2], axis=2, origin=origin)
                        third_filtering = downsample_coeffs(third_filtering, scale)/(self.density_integral * (scale**coeffs.ndim))
                        hessian_elements.append(third_filtering)
                        
        else:
            
            def recursion_for(ind, n_filtering):
            
                if ind<self.ndim-1:
                    for filter_i in bbs_filters:
                        m_filtering = nd.convolve1d(n_filtering, filter_i, mode=self.bcs[ind], axis=ind, origin=origin)
                        recursion_for(ind+1, m_filtering)
                else:
                    for filter_k in bbs_filters:
                            m_filtering = nd.convolve1d(n_filtering, filter_k, mode=self.bcs[ind], axis=ind, origin=origin)
                            m_filtering = downsample_coeffs(m_filtering, scale)/(self.density_integral * (scale**coeffs.ndim))
                            hessian_elements.append(m_filtering)
        
            recursion_for(0, self.values_on_grid.copy())
            
        return np.array(hessian_elements)

    def update_lip(self, *args):#, approx=False, **kwargs):
        self.lip = self.hessian_norm_approx(*args, **self.u_kwargs) if self.approx else self.hessian_norm(*args, **self.u_kwargs)
        return self.u_multi*self.lip


class neg_rdspline_likelihood(rdspline_likelihood):
    ct = -1.
    # def __init__(self,  *args, **kwargs):
    #     super().__init__(self,  *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.ct * super().__call__(*args, **kwargs)

    def gradient(self, *args, **kwargs):
        return self.ct * super().gradient(*args, **kwargs)
    
    jacobianT = gradient

    def hessian(self, *args, **kwargs):
        return self.ct * super().hessian(*args, **kwargs)


if __name__ == '__main__':

    a = np.identity(3)+np.rot90(np.identity(3))
    bcs=["wrap","wrap"]

    c= np.pad(upsample_coeffs(a, 2), 4)
    plt.imshow(c)
    bs_filters = np.array(a.ndim*[bs_to_filter(3)])
    plt.imshow(convolve_coeffs(c, bs_filters, bcs))
    plt.imshow(eval_bspline_on_subgrid(c,3,1,bcs))
    plt.imshow(eval_bspline_on_subgrid(c,3,8,bcs))

    b= np.pad(a, 4)
    plt.imshow(b) #coeffs
    plt.imshow(eval_bspline_on_subgrid(b,3,1,bcs)) #image
    plt.imshow(eval_bspline_on_subgrid(b,3,2,bcs)) #image with better resolution
    plt.imshow(eval_rdspline_on_subgrid(b,3,2,bcs))
    plt.plot(eval_bspline_on_subgrid(b,3,2,bcs)[10,:])
    plt.plot(b[5,:])

    s = 10
    np.sum(np.exp(eval_bspline_on_subgrid(b,3,s,bcs)))/(s**2)
    rdspline_integral(b,3,s,bcs)
    plot_rdspline_integral_convergence(b,3,bcs, scales=9)
    rdspline_integral(b,3,2**7,bcs)

    plt.imshow(rdspline_betaintegral_vector(1, b, 3, 2, bcs, measure=1))

    density_integral, values_on_grid = rdspline_integral(b, 3, 2, bcs, measure=1., out=True)
    plt.imshow(values_on_grid)
    beta_integral = eval_bspline_on_grid(values_on_grid, 3, bcs, scale=2)
    plt.imshow(beta_integral)
    h = -downsample_coeffs(beta_integral, 2)/density_integral
    plt.imshow(h)
    plt.imshow(rdspline_betaintegral_vector(1, b, 3, 2, bcs, measure=1))

    # Checking spline integral
    d = 3
    s = 40

    huhu = np.zeros((5,5))
    huhu[2,2]=1.
    _, v = spline_integral(huhu, d, s, bcs, out=True)
    np.sum(v*v)/(s**huhu.ndim)

    huhu = np.zeros(5)
    huhu[2]=1.
    _, v = spline_integral(huhu, d, s, ["wrap"], out=True)
    (np.sum(v*v)/(s**huhu.ndim))**2


    # Gradient test 
    s=8
    rand_vector = np.random.rand(*b.shape)
    #rand_vector = np.pad(np.random.rand(*a.shape), 4) # no boundary perturbation
    rand_vector /= np.sqrt(np.sum(rand_vector*rand_vector))

    # First part of test
    measure = 1 #np.random.rand(*[s*i for i in b.shape]) + 0.1
    density_integral, values_on_grid = rdspline_integral(b, 3, s, bcs, measure=measure, out=True)
    beta_integral = eval_bspline_on_grid(values_on_grid, 3, bcs, scale=s)/(s**2)
    h = downsample_coeffs(beta_integral, s)/density_integral
    dir_grad = np.sum(h*rand_vector)

    eps = 0.000000001
    (np.log(rdspline_integral(b+eps*rand_vector,3,s,bcs, measure=measure))-np.log(rdspline_integral(b,3,s,bcs, measure=measure)))/eps

    err = []
    epses = []
    for i in range(1, 30):
        eps = 2**(-i)
        fdif = (np.log(rdspline_integral(b+eps*rand_vector,3,s,bcs, measure=measure))-np.log(rdspline_integral(b,3,s,bcs, measure=measure)))/eps
        err.append(np.abs(dir_grad-fdif))
        epses.append(eps)

    plt.loglog(epses, np.array(err)); plt.show()
