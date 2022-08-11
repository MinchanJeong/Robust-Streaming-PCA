import numpy as np
from numba import jit

class stream():
    
    def __init__(self, k, testwindow, clip = False):
        
        self.stream = np.load(f'./datasets/sp500.npy')
        if clip:
            self.stream = np.clip(self.stream, -1.0, 1.0)
        
        self.p = self.stream.shape[0]
        self.T = self.stream.shape[1]
        
        self.k = k
        
        assert testwindow > 0 and testwindow <= self.stream.shape[1]
        self.testwindow = testwindow
        
        U_obj = (self.stream[:,-1*self.testwindow:] @ self.stream[:,-1*self.testwindow:].T) / self.testwindow
        self.U_obj, s, _ = np.linalg.svd(U_obj)
        
    def _generate_xt(self):
        
        x_t = np.copy(self.stream[:,self.count]).reshape(-1,1)
        self.count += 1
        
        if self.count > self.T:
            print('StreamError')
        
        return x_t
    
    def _get_testsp(self ,copy=True):
        
        if copy:
            return np.copy(self.U_obj)
        else:
            return self.U_obj
        
    def _reset(self, adjust):
        
        self.count = self.T % adjust if adjust != 0 else 0
        
        return self.T - adjust, self.p