import math, random, argparse
from collections import deque
from tqdm import tqdm
import numpy as np

from numba import jit
import warnings
warnings.filterwarnings("ignore")

from generate_At import SVD_base_exact

#### Hyperparameters #####################################################
T, T_save, T_oja  = 48000, 10, 5
assert T % T_save == 0
assert T_save % T_oja == 0
k = 5
p = 100

B_list = [1,2,3,4,5,8,10,20,30,50,75,100,150,200,300,500,600,800,1000,1200,1500,2000]
zeta_list = 0.0005 * np.power(1.8, np.arange(22))

## Early Stopping #####
use_earlystop = None
stdev_thres = 0.001
average_num = 100
retain_num = 100

#### Argparse ############################################################
sigma = 1.0
delta = 1.0
gamma = 0.5
loop = 30

parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('-algo', type=str, nargs='?', default = 'oja', help='noisy or oja(str)')
parser.add_argument('-p', type=int, nargs='?', default = p, help='totaldim')
parser.add_argument('-k', type=int, nargs='?', default = k, help='latentdim')
parser.add_argument('-sigma', type=float, nargs='?', default = sigma, help='sigma(float)')
parser.add_argument('-delta', type=float, nargs='?', default = delta, help='delta(float)')
parser.add_argument('-gamma', type=float, nargs='?', default = gamma, help='gamma(float)')
parser.add_argument('-loop', type=int, nargs='?', default = loop, help='loop(int)')
parser.add_argument('-label', type=str, nargs='?', default = '', help='pickle tail(str)')
#parser.add_argument('-earlystop', action='store_true', help='set earlystopping')
args = parser.parse_args()

algo = args.algo
p, k  = args.p, args.k
sigma  = args.sigma 
delta  = args.delta 
gamma  = args.gamma 
loop   = args.loop 
savelabel = args.label
use_earlystop = args.earlystop

#### Functions ###########################################################
def print_status():
    print('\n',args)
    
@jit(nopython=True)
def get_dist(U,Q):
    dist = np.linalg.norm(U[:,k:].T @ Q, ord=2)
    
    return dist

class AvgQueue():
    
    def __init__(self, use_earlystop):
        self.avgqueue = deque(maxlen=retain_num)
        self.use_earlystop = use_earlystop
        self.break_flag = False
        
    def _update(self, distance_list):
        if len(distance_list) > average_num:
            avg = np.average(distance_list[-average_num:])
            self.avgqueue.append(avg)
        
    def _check_earlystop(self, distance_list, t):
        if not self.use_earlystop: return False
        
        self._update(distance_list)
        if len(self.avgqueue) == retain_num:
            stdev = np.std(self.avgqueue)
            self.break_flag = True if (stdev < stdev_thres and t > T//10 ) else False
            
        return self.break_flag
    
#### Algorithms ##########################################################
@jit(nopython=True)
def oja_iter(Q, x_t, zeta, t):
    Q = Q + zeta * x_t @ x_t.T @ Q
    if (t+1) % T_oja == 0:
        Q, _ = np.linalg.qr(Q)
    
    return Q

@jit(nopython=True)
def noisy_power_iter(Q, x_t, B, t):
    Q = Q + (1.0 / B) * x_t @ x_t.T @ Q
    if t == B-1:
        Q, _ = np.linalg.qr(Q)
    
    return Q

def oja_PCA(distance_list, init_Q, zeta):
    Q = init_Q
    avgqueue = AvgQueue(use_earlystop)
    for l in range(T//T_save):
        
        for t in range(T_save):
            x_t = SVD_base_._generate_xt()
            Q = oja_iter(Q, x_t, zeta, t)
            
        U = SVD_base_._get_orthosp(copy=True)
        dist = get_dist(U, Q)
        distance_list.append(dist)
        
        flag = avgqueue._check_earlystop(distance_list, (l+1) * T_save)
        if flag: break
        
    return distance_list, flag

def noisy_power_PCA(distance_list, init_Q, B):
    Q = init_Q
    avgqueue = AvgQueue(use_earlystop)
    for l in range(T//B):
        
        for t in range(B):
            x_t = SVD_base_._generate_xt()
            Q = noisy_power_iter(Q, x_t, B, t)
        
        U = SVD_base_._get_orthosp(copy=True)
        dist = get_dist(U, Q)
        distance_list.append(dist)
        
        flag = avgqueue._check_earlystop(distance_list, (l+1) * B)
        if flag: break
        
    return distance_list, flag
        
#### Set Algorithm #######################################################

if algo == 'oja':
    run_PCA = oja_PCA
    learning_param_list, lparam_str = zeta_list, 'zeta'
    
elif algo == 'noisy':
    run_PCA = noisy_power_PCA
    learning_param_list, lparam_str = B_list, 'B'
    
else:
    raise NotImplementedError

#### Main ###########################################################
print_status()    
datadict_list = []

str_earlystop = '1st iter'
loopbar = tqdm(range(loop))
for loopno in loopbar:    
    datadict = {}
    for lparam_idx, lparam in enumerate(learning_param_list):
        # Initialization
        SVD_base_ = SVD_base_exact(p,k,sigma,delta,gamma)
        U = SVD_base_._get_orthosp(copy=False)
        M = np.random.randn(p, k)
        Q, _ = np.linalg.qr(M)
        
        # Define distance array
        dist_list  = [get_dist(U, Q)]
        
        # main
        loopbar.set_description('Samples:({:d} / {:d}) | {}:({:d} / {:d}) | early stopped:({}) '.format(\
            loopno+1,loop,lparam_str,lparam_idx+1,len(learning_param_list),str_earlystop))
        
        datadict[lparam], flag_earlystop = run_PCA(dist_list, Q, lparam)
        str_earlystop = 'O' if flag_earlystop else 'X'
        
    datadict_list.append(datadict)

#### SAVE ###############################################################
import os, datetime, pickle

savepath = './data/{}'.format(savelabel)
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('New directory was created.')

now = datetime.datetime.now()
formattedDate = now.strftime("%f")
savedir = savepath+'/{}.pickle'.format(formattedDate)

hyperdict = {'T':T, 'T_save':T_save, 'loop':loop, 'k':k, 'p':p, 'sigma':sigma,\
            'gamma':gamma, 'delta':delta, 'use_earlystop':use_earlystop,\
            'stdev_thres':stdev_thres, 'retain_num':retain_num, 'average_num':average_num,\
            'algo':algo, 'lp_list':learning_param_list, 'label':savelabel}
with open(savedir, 'wb') as f:
    pickle.dump(hyperdict, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(datadict_list, f, pickle.HIGHEST_PROTOCOL)