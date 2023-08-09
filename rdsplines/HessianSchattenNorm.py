from typing import Tuple, Optional, Union
from numbers import Number
from rdsplines import array_lib as np


class LpSchattenNorm:
    r"""
    Class for Lp Schatten Norm (i.e. the sum of the lp norm of the singular values
    of all matrices (one per point in the n-dim domain)).
    The proximity operator is computed by sv-decomposing, projecting the svs onto
    the conjugate ball and sv-recomposing with the original basis (except for p=2).
    """

    def __init__(self, hess_shape: Tuple[int, ...], p: Number = 1, hermitian: bool = False, flat: bool = False, flat_input: bool = False):
        r"""
        Parameters
        ----------
        shape: tuple of ints
            Input dimension
        p: int or float or np.infty
            Order of the norm
        hermitian: bool
        flat: bool
            Whether the matrices are flattened (only for Hermitian)
        """

        self.hess_shape = hess_shape
        self.flat_input = flat_input
        self.p = p
        #with np.errstate(divide='ignore'): self.q = 1/(1-1/np.float64(p)) # 1/p+1/q=1  #q=p/(np.float64(p)-1) fails at np.infty
        self.q = np.infty if self.p==1 else 1/(1-1/p)
        self.proj_lq_ball = self.init_proj_lq_ball(self.q)
        self.flat = flat
        self.hermitian = hermitian if not flat else True
        
        if flat:
            self.d2or3 = 2 if hess_shape[-1]==3 else 3 if hess_shape[-1]==6 else None
        else:
            self.d2or3 = 2 if hess_shape[-2:]==(2,2) else 3 if hess_shape[-2:]==(3,3) else None
        
        self.closed_formula = ((self.d2or3 is not None) and hermitian)

        self.flat_index_2d = (np.s_[..., 0], np.s_[..., 1], np.s_[..., 1], np.s_[..., -1], np.s_[:-1])
        self.erect_index_2d = (np.s_[...,0,0], np.s_[...,0,1], np.s_[...,1,0], np.s_[...,1,1], np.s_[:-2])
        self.eps = 2.*np.finfo(float).eps

        self.input_size = np.prod(np.array(hess_shape)) # pycsou flattening
        # output_size = 1 # pycsou flattening
        self.is_differentiable = False
        self.is_linear = False

    def __call__(self, hess: Union[Number, np.ndarray]) -> Number:
        # hess is array with last one/two (flat/not-flat) dimensions being the matrix
        if self.flat_input: hess = hess.reshape(self.hess_shape)

        if not self.p==2:
            singvals = self.svd(hess, compute_basis = False, hermitian = self.hermitian,
                                 flat = self.flat, closed_formula = self.closed_formula)
            s_norm = np.sum(np.linalg.norm(singvals, ord=self.p, axis=-1), axis=None)

        else: # no need for svd because the norm is Frobenius'
            s_norm = np.sum(self.frobenius(hess, self.flat), axis=None)

        return s_norm


    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        # Proximal projects the svals on the lq ball and reconstructs with the svecs
        if self.flat_input: x = x.reshape(self.hess_shape)

        if not self.p==2:
            Uvecs, singvals, Vvecs = self.svd(x, compute_basis = True, hermitian = self.hermitian,
                                            flat = self.flat, closed_formula = self.closed_formula)

            singvals -= tau*self.proj_lq_ball(singvals/tau, radius=1) #(singvals/tau, radius=1)
            prox_x = self.rebuild_svd(Uvecs, singvals, Vvecs,
                                            flat = self.flat, closed_formula = self.closed_formula)

        else: # no need for svd because the norm is Frobenius'
            prox_x = x-tau*self.proj_lq_ball(x/tau, self.flat, radius=1)

        return prox_x.ravel() if self.flat_input else prox_x


    def project(self, x: Union[Number, np.ndarray], radius: Number) -> Union[Number, np.ndarray]:
        # projection only. should not repeat code but not calling project in prox to save some subtractions.
        if self.flat_input: x = x.reshape(self.hess_shape)

        if not self.p==2:
            Uvecs, singvals, Vvecs = self.svd(x, compute_basis = True, hermitian = self.hermitian,
                                            flat = self.flat, closed_formula = self.closed_formula)

            singvals = self.proj_lq_ball(singvals, radius=radius) #(singvals/tau, radius=1)
            project_x = self.rebuild_svd(Uvecs, singvals, Vvecs,
                                            flat = self.flat, closed_formula = self.closed_formula)

        else: # no need for svd because the norm is Frobenius'
            project_x = self.proj_lq_ball(x, self.flat, radius=radius)

        return project_x.ravel() if self.flat_input else project_x


    def svd(self, mat_tensor, compute_basis = True, hermitian = False, flat = False, closed_formula = False):

        if closed_formula:
            if self.d2or3==2:
                return self.svd_2d(mat_tensor, compute_basis = compute_basis, flat=flat)
            else:
                return self.svd_3d(mat_tensor, compute_basis = compute_basis, flat=flat)

        else:
            if flat: mat_tensor = self.erect_symmetric_matrix(mat_tensor)
            return np.linalg.svd(mat_tensor, compute_uv = compute_basis) #cupy:, hermitian = hermitian)


    def rebuild_svd(self, U, s, V, flat = False, closed_formula = False):

        if closed_formula:
            if self.d2or3==2:
                rebuilt = self.rebuild_svd_2d(U, s, V)
            else:
                rebuilt = self.rebuild_svd_3d(U, s, V)
            return rebuilt if flat else self.erect_symmetric_matrix(rebuilt)

        else:
            rebuilt = (U[..., :, :]*s[..., np.newaxis, :]) @ V
            return rebuilt if not flat else self.deflate_symmetric_matrix(rebuilt)
            # (U*s) @ V; #U @ np.diag(s) @ V


    def svd_2d(self, mat_tensor, compute_basis = True, flat = False):
        # closed formula for 2x2 matrices
        # changed the function to hermitian only
        ii, ij, ji, jj, sz = self.flat_index_2d if flat else self.erect_index_2d
        shape_out = (*mat_tensor.shape[sz], 2) # hermit output shape

        tr = mat_tensor[ii] + mat_tensor[jj]
        dt = (mat_tensor[ii] - mat_tensor[jj])**2 + 4*mat_tensor[ij]*mat_tensor[ji]
        # simplified from tr**2 - 4*( mat_tensor[ii]*mat_tensor[jj] - mat_tensor[ij]*mat_tensor[ji])
        singvals = np.zeros(shape_out)
        singvals[..., 0] = 0.5*(tr+np.sqrt(dt))
        singvals[..., 1] = 0.5*(tr-np.sqrt(dt))

        if compute_basis:
            singvec = np.zeros(shape_out)
            zero_ji = (np.abs(mat_tensor[ji])<self.eps).astype(int)
            singvec[..., 0] = (1-zero_ji)*(singvals[..., 0] - mat_tensor[jj])
            singvec[..., 1] = (1-zero_ji)*mat_tensor[ji] # from generic 2x2: e_1,y = e_2,y = ij

            norm = np.sqrt((singvals[..., 0] - mat_tensor[jj])**2 + mat_tensor[ji]**2) + zero_ji
            singvec[..., 0] = singvec[..., 0]/norm + zero_ji
            singvec[..., 1] /= norm
            return None, singvals, singvec
            # Hermitian only returns one singvec because the other can be computed from the first

        return singvals


    def rebuild_svd_2d(self, U, s, V): # for 2x2 hermitian only, so now U is not required (passed as None)

        mat_tensor = np.zeros((*s.shape[:-1], 3))
        # if symmetric then eigenvecs are orthonormal so inverse of change of basis is just the transpose
        # also orthonormal in 2x2 means e_1,x = e_2,y, e_1,y = -e_2,x
        
        mat_tensor[..., 0] = s[..., 0]*V[...,0]**2 + s[..., 1]*V[...,1]**2
        mat_tensor[..., 1] = (s[..., 0]-s[..., 1])*V[..., 0]*V[..., 1]
        mat_tensor[..., 2] = s[..., 0]*V[..., 1]**2 + s[..., 1]*V[...,0]**2

        return mat_tensor
    
    # Onlc need to change the closed formula stuff in svd, rebuild_svd, and init
    def svd_3d(self, mat_tensor, compute_basis = True, flat = False, tol=None):
        if tol is None: tol=self.eps
        # closed formula for 3x3 real symmetric matrices (complex hermitian should be possible too). 
        # assuming flat
        shape_out = (*mat_tensor.shape[:-1], 3) # hermit output shape
        be_prudent = True
        
        a, b, c, d, e, f = mat_tensor[...,0], mat_tensor[...,2], mat_tensor[...,5], mat_tensor[...,1], mat_tensor[...,4], mat_tensor[...,3]
        
        tr = a + b + c
        y = a*a + b*b + c*c - a*b -a*c -b*c + 3*(d*d + f*f + e*e)
        z = -(2*a-b-c)*(2*b-a-c)*(2*c-a-b) \
            + 9*( (2*a-b-c)*e*e + (2*b-a-c)*f*f + (2*c-a-b)*d*d  ) \
            - 54*d*e*f
        
        sqy = np.sqrt(y)
        phi = (np.pi/2)*np.ones_like(a)
        
        phi[z!=0] = np.arctan(np.sqrt(np.maximum(4*y[z!=0]**3 - z[z!=0]**2,0))/z[z!=0])
        phi[z<0] += np.pi
        
        singvals = np.zeros(shape_out)
        singvals[..., 1] = (tr - 2*sqy*np.cos(phi/3) )/3
        singvals[..., 2] = (tr + 2*sqy*np.cos((phi-np.pi)/3) )/3
        singvals[..., 0] = (tr + 2*sqy*np.cos((phi+np.pi)/3) )/3
            
        if compute_basis:
                        
            singvecs = np.zeros((*shape_out,3))
        
            fis0  = np.abs(f)<tol;

            #with np.errstate(divide='ignore', invalid='ignore'):
            for i in range(3): # for the three eigenvectors
            
                m = np.where(fis0,
                                    (f*f - (c-singvals[..., i])*(a-singvals[..., i]))/(e*(a-singvals[..., i])-d*f),
                                    (d*(c-singvals[..., i])-e*f)/(f*(b-singvals[..., i])-d*e)
                                    )
            
                singvecs[..., i, 2] = 1
                
                singvecs[..., i, 1] = m
                
                singvecs[..., i, 0] = np.where(fis0,  (m*(singvals[..., i]-b) - e)/d, (singvals[..., i] - c - e*m)/f)                        
        
            if be_prudent:
                
                dis0 = np.abs(d)<tol; eis0 = np.abs(e)<tol
                def0 = dis0 & fis0 & eis0; defnot0 = ~def0
                df0 = dis0 & fis0 & defnot0
                de0 = dis0 & eis0 & defnot0
                ef0 = eis0 & fis0 & defnot0
                
                # put all potential nans to zero
                singvecs[df0 | de0 | ef0 | def0, ...] = 0
                
                # already diag
                singvecs[def0, 0, 0] = singvecs[def0, 1, 1] = singvecs[def0, 2, 2] = 1.
                
                # fill the one already-diagonal vector in semidiag matrices
                singvecs[df0, 0, 0] =  singvecs[de0, 1, 1] =  singvecs[ef0, 2, 2] = 1.
                
                # first vector is diagonalized (df0)
                # should i just call svd_2d? whatev
                singvecs[df0, 1, 1] = -e[df0]
                singvecs[df0, 1, 2] = b[df0]-singvals[df0, 1]
                singvecs[df0, 2, 1] = -singvecs[df0, 1, 2]
                singvecs[df0, 2, 2] = singvecs[df0, 1, 1]
                
                # second
                singvecs[de0, 0, 0] = -f[de0]
                singvecs[de0, 0, 2] = a[de0]-singvals[de0, 1]
                singvecs[de0, 2, 0] = -singvecs[de0, 0, 2]
                singvecs[de0, 2, 2] = singvecs[de0, 0, 0]
                singvals[de0] = np.roll(singvals[de0],1)[:,::-1]
                #not sure taking like this is general or i have to just recompute the singvals (eigenvals)
                
                #third
                singvecs[ef0, 1, 0] = -d[ef0]
                singvecs[ef0, 1, 1] = a[ef0]-singvals[ef0, 1]
                singvecs[ef0, 0, 0] = -singvecs[ef0, 1, 1]
                singvecs[ef0, 0, 1] = singvecs[ef0, 1, 0]
                
                     
            singvecs = singvecs/np.linalg.norm(singvecs, axis=2)[:,:,np.newaxis]
                    
            return None, singvals, singvecs
                
        return singvals
    
    
                
    
    def rebuild_svd_3d(self, U, s, V):
        
        # should i just compute the third by crossproduct?
        mat_tensor = np.zeros((*s.shape[:-1], 6))

        # Fill diagonal first [..., m,n], m is eigenvector index, n is component
        mat_tensor[...,0] = s[..., 0]*V[..., 0, 0]**2 + s[..., 1]*V[..., 1, 0]**2 + s[..., 2]*V[..., 2, 0]**2
        mat_tensor[...,2] = s[..., 0]*V[..., 0, 1]**2 + s[..., 1]*V[..., 1, 1]**2 + s[..., 2]*V[..., 2, 1]**2
        mat_tensor[...,5] = s[..., 0]*V[..., 0, 2]**2 + s[..., 1]*V[..., 1, 2]**2 + s[..., 2]*V[..., 2, 2]**2
        
        # 1off diagonal
        mat_tensor[...,1] = s[..., 0]*V[..., 0, 0]*V[..., 0, 1] + s[..., 1]*V[..., 1, 0]*V[..., 1, 1] + s[..., 2]*V[..., 2, 0]*V[..., 2, 1]
        mat_tensor[...,4] = s[..., 0]*V[..., 0, 2]*V[..., 0, 1] + s[..., 1]*V[..., 1, 2]*V[..., 1, 1] + s[..., 2]*V[..., 2, 2]*V[..., 2, 1]
        
        # 2off diagonal
        mat_tensor[...,3] = s[..., 0]*V[..., 0, 0]*V[..., 0, 2] + s[..., 1]*V[..., 1, 0]*V[..., 1, 2] + s[..., 2]*V[..., 2, 0]*V[..., 2, 2]
        
        # Wolfram input for multiplication (not same naming convention as elsewhere)
        #[[a, b, c],[d, e, f],[g, h, i]]* [[j, 0, 0],[0, k, 0],[0, 0, l]] *[[a, d, g],[b, e, h],[c, f, i]]
        
        #if real eigens and orthogonal should be symmetric (?)
        
        return mat_tensor


    def erect_symmetric_matrix(self, flat_mat):
        # from (x,...,z, m12) to (x,...,z,m1,m2) when symmetric
        *shape, deg_free = flat_mat.shape
        dim = int(np.sqrt(deg_free*2))
        assert(dim*(dim+1)/2 == deg_free), "degrees of freedom do not match dimensions"

        row, col = np.triu_indices(dim)

        erect_mat = np.zeros((*shape, dim, dim), dtype=flat_mat.dtype)
        erect_mat[..., col, row] = np.conjugate(flat_mat) # first lower (order does not matter because diagonal is real)
        erect_mat[..., row, col] = flat_mat # then upper
        return erect_mat


    def deflate_symmetric_matrix(self, erect_mat):
        # from (x,...,z, m1,m2) to (x,...,z, m12) when symmetric
        *shape, dim, dim = erect_mat.shape
        deg_free = int(dim*(dim+1)/2)

        row, col = np.triu_indices(dim)

        flat_mat = np.zeros((*shape, deg_free), dtype=erect_mat.dtype)
        flat_mat[..., :] = erect_mat[..., row, col]
        return flat_mat


    def frobenius(self, x, flat):
        if flat:
            *shape, deg_free = x.shape
            dim = int(np.sqrt(deg_free*2))
            diag_inds = np.cumsum(np.arange(dim+1,1,-1))-dim-1
            x2 = x**2
            return np.sqrt(2*np.sum(x2, axis=-1) - np.sum(x2[..., diag_inds], axis=-1))
        else:
            return np.sqrt(np.einsum('...ijk,...ijk->...i', x, x))


    def prox_l2(self, x, flat, tau):
        Frob = self.frobenius(x, flat)
        slc = np.s_[tuple([...]+(2-flat)*[np.newaxis])]
        with np.errstate(divide='ignore'):
            return np.maximum(1-tau/Frob, 0)[slc]*x


    def proj_l2_ball(self, x, flat, radius):
        Frob = self.frobenius(x, flat)
        slc = np.s_[tuple([...]+(2-flat)*[np.newaxis])]
        if Frob <= radius:
            return x
        else:
            return radius*x/Frob[slc]

    def proj_linfty_ball(self, x, radius):
        return np.clip(x, a_min=-radius, a_max=radius)

    def proj_l1_ball(self, x, radius):
        raise NotImplementedError("l1 ball projection is not implemented") # any brent for cupy? scipy.opt not ported

    def init_proj_lq_ball(self, q):
        assert (q in [np.inf, 1, 2]), "as of now, the p-norm proximal is only supported for q = 1, 2, inf" # in the Schatten context

        if q == np.inf:
            return self.proj_linfty_ball
        elif q == 1:
            return self.proj_l1_ball
        elif q == 2:
            return self.proj_l2_ball # do not take main proj_l2, adapt to Schatten


class HessianSchattenNorm:
    r"""
    Class for Hessian Schatten Norm.
    """

    def __init__(self, Hess, SchattenNorm, n_prox_iter, lam=1., iter_func=None, indicator_project=None, indicator_norm=None, hess_ct=None):
        r"""
        Parameters
        ----------
        Hess: instance of a Hessian class
            acting as a linear operator (with .__call__ and .adjoint and .dim)
        SchattenNorm: instance of the LpSchattenNorm class
        n_prox_iter: int
            Number of iterations for the prox computation
        iter_func: function
            Any rule to update n_prox_iter every time it is called
        indicator_project:
            Projector related to desired indicator function
        indicator_norm:
            Norm of the above
        """
        self.Hess = Hess
        self.SchattenNorm = SchattenNorm
        self.coeffs_shape = Hess.coeffs_shape
        self.hess_shape = Hess.hess_shape
        self.hess_dim = np.prod(np.array(self.hess_shape))
        self.input_size = np.prod(np.array(self.coeffs_shape))

        self.n_prox_iter = n_prox_iter
        self.lam = lam
        if iter_func == None: self.iter_func = lambda x: x
        if indicator_project == None: self.indicator_project = lambda x: x
        if indicator_norm == None: self.indicator_norm = lambda x: 0
        if hess_ct == None: self.hess_ct = 16*(self.Hess.dim**2)

        self.is_differentiable = False
        self.is_linear = False

    def __call__(self, coeffs: Union[Number, np.ndarray]) -> Number:

        return self.lam*(self.SchattenNorm(self.Hess(coeffs)) + self.indicator_norm(coeffs))


    def prox(self, z: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        tau = self.lam*tau

        self.n_prox_iter = self.iter_func(self.n_prox_iter)
        ct = 1./(self.hess_ct*tau)
        i=0
        Psi = Omega_prev = np.zeros(self.hess_shape) #np.zeros(self.hess_dim)
        t_prev = 1

        while i<self.n_prox_iter:

            Omega = self.SchattenNorm.project(
                    Psi + ct*self.Hess(
                    self.indicator_project(
                    z-tau*self.Hess.adjoint(Psi))), radius=1)
            t = 0.5*(1 + np.sqrt(1+4*t_prev**2))
            Psi = Omega + (Omega - Omega_prev)*(t_prev-1)/t

            i=i+1
            t_prev = t
            Omega_prev = Omega

        return self.indicator_project(z-tau*self.Hess.adjoint(Omega))


if __name__ == '__main__':


    # Testing slicing bc_behavior
    slc= np.s_[..., 0]
    np.array([[  0,   1,   2], [  3,   4,   5],[  6,   7,   8]])[slc]

    ## Testing
    hs_shape = (4,6,3)
    np.array([1,0,1])

    lsn = LpSchattenNorm(hs_shape, p=1, hermitian=True, flat=True)
    lsn2 = LpSchattenNorm(hs_shape, p=2, hermitian=True, flat=True)
    lsnf = LpSchattenNorm(hs_shape, p=np.infty, hermitian=True, flat=True)

    hess_id = np.array(4*6*[1,0,1]).reshape(hs_shape) #1,1
    lsn(hess_id)/2
    lsn2(hess_id)/np.sqrt(2)
    lsnf(hess_id)
    hess_id = 2.*np.array(4*6*[1,0,1]).reshape(hs_shape) #1,1
    lsn(hess_id)/2
    lsn2(hess_id)/np.sqrt(2)
    lsnf(hess_id)
    other = np.array(4*6*[2,3,2]).reshape(hs_shape) #5,-1
    lsn(other)/(5+1)
    lsn2(other)/np.sqrt(5**2+1)
    lsnf(other)/5
    other = np.array(4*6*[8,8,8]).reshape(hs_shape) #16,0
    lsn(other)/16
    lsn2(other)/16
    lsnf(other)/16
    other = np.array(4*3*[8,8,8]+4*3*[2,3,2]).reshape(hs_shape) #16,0
    lsn(other)/((5+1+16)/2)
    lsn2(other)/((16+np.sqrt(5**2+1))/2)
    lsnf(other)/((16+5)/2)


    ## Testing cases
    # flat and closed
    #import timeit #timeit.timeit("
    import time
    leeway = 30.
    eps = 1e-8
    x, y = 2*[2] # at 10000 the timings are 20s,41s,104s,110s
    m = 3
    a1 = np.arange(x*y*m).reshape(x,y,m) # err grows with matrix size

    # np.linalg.norm(a2, ord=2, axis=(-2,-1)) # not expected behavior must code explicitly

    # not-flat and closed
    start = time.time()
    u,s,v = lsn.svd_2d(a1, compute_basis = True, flat = True)
    err = lsn.rebuild_svd_2d(u,s,v)-a1
    np.max(err), np.mean(np.abs(err))
    np.max(err)<leeway*eps
    time.time()-start
    n = np.linalg.norm(s, ord=1, axis=-1)
    n, np.sum(n,axis=None)

    # not-flat and closed
    start = time.time()
    a1e = lsn.erect_symmetric_matrix(a1)
    u,s,v = lsn.svd(a1e, compute_basis = True, hermitian = True, flat = False, closed_formula = True)
    err = a1e-lsn.rebuild_svd(u, s, v, flat = False, closed_formula = True)
    np.max(err), np.mean(np.abs(err))
    np.max(err)<leeway*eps
    time.time()-start
    n = np.linalg.norm(s, ord=1, axis=-1)
    n, np.sum(n,axis=None)

    # flat and not-closed
    start = time.time()
    u,s,v = lsn.svd(a1, compute_basis = True, hermitian = True, flat = True, closed_formula = False)
    err = a1-lsn.rebuild_svd(u, s, v, flat = True, closed_formula = False)
    np.max(err), np.mean(np.abs(err))
    np.max(err)<leeway*eps
    time.time()-start
    n = np.linalg.norm(s, ord=1, axis=-1)
    n, np.sum(n,axis=None)

    # not-flat and not-closed
    start = time.time()
    u,s,v = lsn.svd(a1e, compute_basis = True, hermitian = True, flat = False, closed_formula = False)
    err = a1e-lsn.rebuild_svd(u, s, v, flat = False, closed_formula = False)
    np.max(err), np.mean(np.abs(err))
    np.max(err)<leeway*eps
    time.time()-start
    n = np.linalg.norm(s, ord=1, axis=-1)
    n, np.sum(n,axis=None)
    
    from bs_Hessian import bs_Hessian
    from bs_utils import bs_deriv_filter, bs_to_filter
    deg = 3
    deriv_order = 2 # Hessian
    bs_filters = [bs_deriv_filter(i) for i in range(deriv_order+1)]
    bs_filters[0] = bs_to_filter(deg, int_grid=True)
    bs_filters[1]= np.array([1., 0 ,-1.])/2.
    coeffs_shape =(5,8)

    Hessian = bs_Hessian(coeffs_shape, bs_filters, ["wrap", "wrap"])
    Schatt_Norm = LpSchattenNorm(Hessian.hess_shape, p=1, hermitian=True, flat=True, flat_input=False)
    Hessian_Schatt_Norm = HessianSchattenNorm(Hessian, Schatt_Norm, 5)

    rand_coeffs = np.random.rand(*coeffs_shape)
    rand_hess = np.random.rand(*Hessian.hess_shape)

    Hessian(rand_coeffs).shape
    Hessian.adjoint(rand_hess).shape
    
    Hessian_Schatt_Norm(rand_coeffs)
    Hessian_Schatt_Norm.prox(rand_coeffs, 1)

    Schatt_Norm(rand_hess)
    Schatt_Norm.prox(rand_hess, 1)
