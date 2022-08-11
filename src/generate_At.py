import math, random, pickle, time
import numpy as np

from numba import jit
import warnings
warnings.filterwarnings("ignore")

#@jit('void(f8[:,:],i8,i8,f8)', nopython=True)
@jit(nopython=True)
def _update_U_numba(U, p, N, theta_max):
    
    rot_idx_list = np.random.choice(np.arange(p), size=2*N, replace=False)

    for n in range(N):
        if n == 0:
            thet = (2*np.random.randint(2)-1) * theta_max
        else:
            thet = np.random.uniform(-1.0,1.0) * theta_max

        cos, sin = np.cos(thet), np.sin(thet)

        i,j = rot_idx_list[2*n], rot_idx_list[2*n+1]
        U_vec1 = cos * U[:,i] - sin * U[:,j]
        U_vec2 = sin * U[:,i] + cos * U[:,j]
        U[:,i] = U_vec1
        U[:,j] = U_vec2
    
    return U

@jit(nopython=True)
def _rotate_U_numba(U, p, N, theta_max):

    cos, sin = np.cos(theta_max), np.sin(theta_max)

    i,j = 0, p - 1
    U_vec1 = cos * U[:,i] - sin * U[:,j]
    U_vec2 = sin * U[:,i] + cos * U[:,j]
    U[:,i] = U_vec1
    U[:,j] = U_vec2

    return U
        
#@jit('f8[:,:](f8[:,:],i8,i8,i8,f8,f8)',nopython=True)
@jit(nopython=True)
def _step_At_numba(U, p, k, N, delta, theta_max):
    
    V = np.random.randn(k, k)
    V, _ = np.linalg.qr(V)

    #SIG_diag = (2.0*np.random.randint(2,size=k)-1.0)
    SIG_diag = 2.0*np.trunc(2.0*np.random.random_sample(size=k))-1.0
    SIG_diag *= np.sqrt(delta)
    SIG = np.zeros((p, k))
    np.fill_diagonal(SIG,SIG_diag)
    U = _update_U_numba(U, p, N, theta_max)

    return (U @ (SIG @ V.T))


#@jit('f8[:,:](f8[:,:],i8,i8,i8,f8,f8)',nopython=True)
@jit(nopython=True)
def _step_At_rotate_numba(U, p, k, N, delta, theta_max):
    
    SIG_diag = np.ones(k) * np.sqrt(delta)
    SIG = np.zeros((p, k))
    np.fill_diagonal(SIG,SIG_diag)
    U = _rotate_U_numba(U, p, N, theta_max)

    return (U @ SIG)


@jit(nopython=True)
def _generate_xt_numba(A_t, p, k, sigma):

    z_t = np.random.randn(k, 1)
    w_t = np.random.randn(p, 1) * sigma
    x_t = A_t @ z_t + w_t

    return x_t
#_________________________________________________________________________
    
class SVD_base():
    
    def __init__(self, p, k, sigma, delta, gamma, N=30, rotate_type=False):
        self.p = p
        self.k = k 
        self.sigma = sigma
        self.delta = delta
        self.gamma = gamma
        
        self.N = N
        self.rotate = rotate_type
        
        U = np.random.randn(self.p, self.p)
        U,_ = np.linalg.qr(U)
        self.U = np.ascontiguousarray(U)
        self.theta_max = math.asin(gamma/delta)
        self.SIG_diag = np.ones(self.k)
        
        # using step At as initialization
        self.A = np.zeros((p,k))
        self._step_At()
        
    def _step_At(self):
        if self.rotate:
            self.A = _step_At_rotate_numba(self.U, self.p, self.k, self.N, self.delta, self.theta_max)
        else:
            self.A = _step_At_numba(self.U, self.p, self.k, self.N, self.delta, self.theta_max)
            
    def _generate_xt(self):
        self._step_At()
        x_t = _generate_xt_numba(self.A, self.p, self.k, self.sigma)
        
        return x_t
    
    def _get_orthosp(self, copy=True):
        
        if copy:
            return np.copy(self.U)
        else:
            return self.U
        
    def _get_At(self, copy=True):
        
        if copy:
            return np.copy(self.A)
        else:
            return self.A
        
    def _reset(self, adjust):
        
        return 0, self.p

'''
#@jit('f8[:,:](f8[:,:],i8,i8,i8,f8,f8)',nopython=True)
@jit(nopython=True)
def _step_At_partialsv_numba(U, p, k, N, delta, theta_max):
    
    V = np.random.randn(k, k)
    V, _ = np.linalg.qr(V)

    SIG_diag = np.asarray([1.0,1.0,1.0,0.7,0.7])
    SIG_diag *= np.sqrt(delta)
    SIG = np.zeros((p, k))
    np.fill_diagonal(SIG,SIG_diag)
    U = _update_U_numba(U, p, N, theta_max)

    return (U @ (SIG @ V.T))
'''