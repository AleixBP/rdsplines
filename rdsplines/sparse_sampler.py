from rdsplines import array_lib as np
from plt import plt
import functools


def project_on_grid(points, grid=None):
    return np.rint(points)


def outter_product_indexing(base_vectors): # one vector per point and per dimension
    dim = len(base_vectors)
    ein_args = (2*dim+1)*[None]
    outter_map = list(range(dim,-1,-1)) #np.arange(dim+1)[::-1]
    if dim>1: outter_map[-1] = 1; outter_map[-2] = 0;
    ein_args[:-1:2] = base_vectors # not a copy, only a pointer
    # ein_args[1::2] = np.stack((np.arange(dim),
    #                       dim*np.ones(dim, dtype=int)), # naming dimensions in base_vectors
    #                       axis=-1)
    ein_args[1::2] = [[i,j] for i,j in zip(range(dim),dim*[dim])]
    ein_args[-1] = outter_map # output form to create outter product

    return ein_args
    # einsum indexing: in 3D it's [0,3], [1,3], [2,3] -> [0,1,2,3], equivalent to 'il, jl, kl->ijkl',
    # to add transpose simply reverse order of axes [0,1,2,3][::-1] or lkji
    # we do 'im, jm, km, lm -> mlkij',


class bs_SparseSampler:

    def __init__(self, coeffs_shape, sample_coords, measuring_grid, basis, support, bcs, sparse_library=False, type = np.float64):
        # Prepare sparse matrix SS that goes from coefficients to an array of the recovered function evaluated at the same place as the samples, ideally we look for SS*coeffs = data
        r"""
        Parameters
        ----------
        size : tuple of ints representing the shape of the input array.
        coeffs_shape : shape of the coefficients
        deriv_filters : list of np.array() filters as [b_spline filter, 1st deriv filter, 2nd deriv filter]
        bcs : list of behavior of the bcs in each axis in relation to the coefficients
        """

        self.sparse_library = sparse_library
        if sparse_library:
            import sparse as sp
        else:
            from rdsplines import scipy_lib; sp = scipy_lib.sparse
        self.sp = sp
        self.bcs = bcs
        self.coeffs_shape = coeffs_shape
        dim = sample_coords.shape[-1]
        self.n_samples = len(sample_coords)
        half_support = support/2
        support_vector = np.arange(-half_support, half_support+1, dtype=int) # assuming the basis is centered at 0, half support each way. assume symmetric so can build Sijk with ::-1
        n_coeffs_involved = (len(support_vector))**dim # (support+1)**dim # gross estimate (adding dim unneeded but sparse constructor will take care of them)

        closest_grid_coords = project_on_grid(sample_coords, grid=measuring_grid)
        dist_to_grid = sample_coords - closest_grid_coords
        nd_support_combinations = np.array(np.meshgrid(*dim*[support_vector])).T.reshape(-1,dim)
        assert(len(nd_support_combinations[...,0]) == n_coeffs_involved)

        # indexing the output
        ii = np.broadcast_to(np.arange(self.n_samples), (n_coeffs_involved, self.n_samples)).T.ravel() # 10xfaster

        # indexing the input coefficients
        self.bc_behavior = self.handle_bcs()
        self.inx = (np.repeat(closest_grid_coords.astype(int).T, n_coeffs_involved, axis=1) 
              + np.tile(nd_support_combinations.T, self.n_samples))
        jjkk = (np.repeat(closest_grid_coords.astype(int).T, n_coeffs_involved, axis=1) 
              + np.tile(nd_support_combinations.T, self.n_samples)) % self.bc_behavior.astype(int)[:,np.newaxis]

        # Exploit separable basis but evaluates b-splines more than necessary (dim*samples*support**dim). #added [::-1]
        # Instead, lets do the minimum (dim*samples*support). Since basis not linear, cannot evaluate before summing. #added [::-1]
        evaluated_support_vector = basis((dist_to_grid[:, np.newaxis]+support_vector[::-1, np.newaxis]).T) # create uniform vectors around samples in all directions according to distance and evaluate
        Sijk = np.einsum(*outter_product_indexing(evaluated_support_vector)).ravel() # outter product by columns gives tensor of evaluations from vector

        # Take out anything outside the domain
        outside_each = jjkk>((np.array(self.coeffs_shape)-1)[:, np.newaxis])
        outside_either = np.any(outside_each, axis=0)
        Sijk[outside_either] = 0.
        
        self.build_sparse(ii, jjkk, Sijk)
        

    def build_sparse(self, ii, jjkk, Vijk):

        if self.sparse_library:
            self.SS = self.sp.COO((ii, *jjkk), Vijk, shape=(self.n_samples,)+self.coeffs_shape)

        else:
            jj = np.ravel_multi_index(jjkk, self.coeffs_shape)
            self.SS = self.sp.coo_matrix((Vijk, (ii, jj)), shape=(self.n_samples,)+ (np.prod(np.array(self.coeffs_shape)),))
            #self.SS.tobsr() # not yet in cupyx
            self.SS.tocsr()

        self.SS_adj = self.SS.transpose()

    def __call__(self, coeffs):
        # input coefficients get the values (1d vector) at the desired points
        return self.sp.tensordot(self.SS, coeffs, axes=coeffs.ndim) if self.sparse_library else self.SS*coeffs.flatten()

    def adjoint(self, evals):
        return self.sp.tensordot(self.SS_adj, evals, axes=1).T if self.sparse_library else (self.SS_adj*evals).reshape(self.coeffs_shape)
        # self.SS_adj*evals or np.dot(SS.T, evals) or SS.H or SS._rmatvec(evals)

    def HTH(self, coeffs):
        return self.adjoint(self.__call__(coeffs))

    def sp_HTH(self, coeffs):
        return self.sp.tensordot(self.SSTSS, coeffs, axes=coeffs.ndim).T

    @property
    @functools.lru_cache()
    def SSTSS(self):
        return self.sp.tensordot(self.SS_adj, self.SS, axes=[-1,0])

    def sum_samples(self, plot=False):
        self.sample_sum = self.SS.sum(axis=0)

        if not self.sparse_library: 
            self.sample_sum = self.sample_sum.reshape(self.coeffs_shape)
            if plot:
                plt.imshow(self.sample_sum)

        elif plot: 
            plt.imshow(self.sample_sum.todense())
        
        return self.sample_sum

    def handle_bcs(self):
        return np.array([domain_len if bc=="wrap" else domain_len*2
                         for domain_len, bc in zip(self.coeffs_shape, self.bcs)], dtype=int)
    ''' Explanation on how bcs work:
    - The domain goes from 0 to coeffs_shape-1
    - Integer indices around the samples are gathered in jjkk: these can be negative and also go beyond the domain
    - But are then trimmed by %bc_behavior.
    - If periodic then its the size of the domain itself.
        (both negativeS and positives go aorund. E.g. with sh=10 total domain is 0 to 9: 10->0, -1->9)
    - Otherwise 2*size so negatives go far away 
    - To deal with the non-periodic we trim any indices bigger than the size at the end by putting the matrix to zero
    - The sparse matrix builder seems to apply a periodic % when shape of matrix is specified. Could only exploit if all bcs are periodic.
    '''


if __name__ == '__main__':
    import bs_utils as bu

    domain_offset = np.array([0.,0.]) # this 4 to change dimension
    domain_length = np.array([10.,10.])
    coeffs_shape = (10, 10)
    bcs = ["wrap", "wrap"]

    n_samples = 5#500
    degree = 3

    domain_dim = len(domain_length)
    sample_coords = domain_offset + domain_length*np.random.rand(n_samples, domain_dim)
    sample_coords = np.array([[4.,4.],[4.,4.],[10.,4.], [9.,4.],[4.,10.],[4.,9.],[10.,10.],[.5,.5]])

    bss = bs_SparseSampler(coeffs_shape, sample_coords, None, lambda args: bu.bs(args, degree), degree+1, bcs)

    for choose_sample in range(len(sample_coords)):
        pl=plt.imshow(bss.SS.todense()[choose_sample].reshape(coeffs_shape)); plt.colorbar(pl); plt.show()

    cos = np.zeros(coeffs_shape); cos[50,50]=1.
    cos = np.random.rand(int(np.prod(np.array(coeffs_shape)))).reshape(coeffs_shape)
    evs = np.random.rand(n_samples)

    np.sum(bss.adjoint(evs)>0.2)
    n_samples*np.prod(coeffs_shape)
    bss.SS.nnz
    bss.SS_adj.nnz
    bss.SSTSS.nnz
    np.max(bss.adjoint(bss(cos))-bss.HTH(cos))
    np.max(bss.adjoint(bss(cos))-bss.sp_HTH(cos))

    bss.SS.density
    bss.SS_adj.density
    bss.SSTSS.density

    a = bss.sum_samples(True)
    n_samples-np.sum(a)
    b=(sample_coords[sample_coords[:,1]>65])
    b=(b[b[:,1]<80])
    pl=plt.imshow(a.todense()[60:65,65:75]); plt.colorbar(pl)
    plt.imshow(a.todense()[59:70,65:75])
    np.sum(a.todense()[59:70,65:75])
    bu.bs(0.63, 3)*bu.bs(0.0, 3)+bu.bs(0.39, 3)*bu.bs(0.5, 3)+bu.bs(0.23, 3)*bu.bs(0.05, 3)
    a.todense()[62,70]