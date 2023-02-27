from rdsplines import array_lib as np
from rdsplines.bs_utils import convolve_coeffs, adjoint_conv_bcs
import itertools as it


def deriv_combinations(dim, deriv_order = 2):
    # could be faster but not expecting high dimensions, otherwise check:
    return np.array([ i for i in it.product(np.arange(deriv_order+1), repeat=dim)
                        if sum(i)==deriv_order])#[::-1]
    # return [ tuple(i) for i in it.product(range(deriv_order+1), repeat=dim)
    #                     if sum(i)==deriv_order]#[::-1]


class bs_Hessian:

    def __init__(self, coeffs_shape, deriv_filters, bcs, flat_input=False):
        # Compute Hessian at uniform grid given the coefficients of a separable basis that can be expressed as a convolution
        r"""
        Parameters
        ----------
        size : tuple of ints representing the shape of the input array.
        shape :
        deriv_filters : list of np.array() filters as [b_spline filter, 1st deriv filter, 2nd deriv filter]
        bcs : list of boundary conditions for the splines as strings ["wrap", "constant", ...]
        """

        self.flat_input = flat_input
        self.coeffs_shape = coeffs_shape
        #self.deriv_filters = np.array(deriv_filters, dtype=object)
        self.deriv_filters = deriv_filters # changed for cupy, added select function
        self.bcs = bcs

        self.dim = len(coeffs_shape) # geometric dimension
        self.deg_free = int(self.dim*(self.dim+1)/2) # degrees of freedom if symmetric
        self.hess_shape = (*self.coeffs_shape, self.deg_free)
        #self.adj_deriv_filters = np.array([filt[::-1] for filt in self.deriv_filters], dtype=object) # do not simply to flip to allow for diff dimensions
        self.adj_deriv_filters = [filt[::-1] for filt in self.deriv_filters] # changed for cupy, added select function
        self.adj_bcs = adjoint_conv_bcs(self.bcs)
        self.d_combinations = deriv_combinations(self.dim, deriv_order = 2) # num of iterations produced should be eq to deg_free

        self.crop = tuple(self.dim*[slice(None)])
        self.slicing = self.deg_free*[tuple(self.dim*[slice(None)])]
                        # depending on the convolution and the length of the kernel
                        # [slice(*((int((max_len - len(filter))/2))*np.array([1,-1]))) for filter in self.deriv_filters]

        self.input_size = np.prod(np.array(coeffs_shape))
        self.output_size = self.deg_free*self.input_size

    def __call__(self, coeffs):
        # Get Hessian from coefficients
        # should check padding make extra dimensions depending on bc? (here or before?)
        # maybe should add slicing?

        if self.flat_input: coeffs = coeffs.reshape(self.coeffs_shape) # pycsou reshape
        hess = np.zeros(self.hess_shape)

        for i, combi in enumerate(self.d_combinations):
            hess[..., i] = convolve_coeffs(coeffs, self.select(self.deriv_filters, combi), self.bcs) # assume filters are separable in each dimension
            # before cupy support: self.deriv_filters[combi] because could set dtype=object (combis are arrays of different sizes)

        return hess.ravel() if self.flat_input else hess

    def adjoint(self, hess):
        # not sure about the boundary conditions of the adjoint
        # if filters are symmetric should be (different dimensions appart) self-adjoint

        if self.flat_input: hess = hess.reshape(self.hess_shape) # pycsou reshape
        coeffs = np.zeros(self.coeffs_shape) # 0th position indexes the deg_free of the matrix

        for i, combi in enumerate(self.d_combinations):
            coeffs[self.slicing[i]] += convolve_coeffs(hess[..., i], self.select(self.adj_deriv_filters, combi), self.adj_bcs)

        return coeffs[self.crop].ravel() if self.flat_input else coeffs[self.crop] # pycsou flatten

    def select(self, arr, indices):
        return [arr[int(i)] for i in indices]


if __name__ == '__main__':
    from .bs_utils import bs_deriv_filter, bs_to_filter, bs
    deg = 3
    deriv_order = 2 # Hessian
    bs_filters = [bs_deriv_filter(i) for i in range(deriv_order+1)]
    bs_filters[0] = bs_to_filter(deg, int_grid=True)
    bs_filters[1]= np.array([1., 0 ,-1.])
    coesp =(5,8)
    coesp_size = int(np.prod(np.array(coesp))) # cupy compatibility
    hesp = coesp + (3,)
    hesp_size = int(np.prod(np.array(hesp))) # cupy compatibility
    
    xyd = np.random.rand(hesp_size).reshape(hesp)
    xyd = (xyd**2 +3.)/100.
    xy = np.random.rand(coesp_size).reshape(coesp)
    xy = (xy**3 +5.)/1000.
    bh = bs_Hessian(coesp, bs_filters, ["wrap", "constant"])
    #bh(xy).reshape(hesp)
    ra = bh.adjoint(xyd).reshape(coesp)
    r = bh(xy).reshape(hesp)

    np.sum(ra*xy)-np.sum(xyd*r)
    bh.d_combinations
