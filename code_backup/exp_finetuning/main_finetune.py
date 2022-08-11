import math, random, argparse, time
import numpy as np

from numba import jit
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/home/mcjeong/jupyter/streamingPCA/src')
from generate_At import SVD_base

DEBUG = False
#### Hyperparameters #####################################################
T = 144000
T_save, T_oja = 100, 5
rotate_type = True

assert T % T_save == 0
assert T_save % T_oja == 0
k = 5
p = 100

B_list = np.asarray([2,3,8,10,20,30,40,60,80,100,200,300,400,600,800,1000,1200,1500,1800,2000,3000,4000,6000,8000,9600])
zeta_list = 1.0 / B_list

#### Argparse ############################################################
sigma = 1.0
delta = 1.0
gamma = 0.00001
loop = 30

parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('-algo', type=str, nargs='?', default = 'oja', help='noisy or oja(str)')
parser.add_argument('-gamma', type=float, nargs='?', default = gamma, help='gamma(float)')
parser.add_argument('-sigma', type=float, nargs='?', default = sigma, help='sigma(float)')
parser.add_argument('-delta', type=float, nargs='?', default = delta, help='delta(float)')
parser.add_argument('-loop', type=int, nargs='?', default = loop, help='loop(int)')
parser.add_argument('-label', type=str, nargs='?', default = '', help='pickle tail(str)')

parser.add_argument('-loweridx', type=int, nargs='?', default = 123)
parser.add_argument('-upperidx', type=int, nargs='?', default = 123)
args = parser.parse_args()

algo = args.algo
sigma  = args.sigma 
delta  = args.delta 
gamma  = args.gamma 
loop   = args.loop 
savelabel = args.label

lower_idx, upper_idx = args.loweridx, args.upperidx

#### Functions ###########################################################
def print_status():
    print('\n',args)
    
# ord='fro' T
@jit(nopython=True)
def get_dist(U,Q):
    dist = np.linalg.norm(U[:,k:].T @ Q,ord=2)
    
    return dist

#### Algorithms ##########################################################
@jit(nopython=True)
def oja_iter(Q, x_t, zeta, t):
    Q = Q + zeta * x_t @ x_t.T @ Q
    if (t+1) % T_oja == 0:
        Q, _ = np.linalg.qr(Q)
    
    return Q

@jit(nopython=True)
def noisy_power_stack(M, B, x_t):
    M = M + (1.0 / B) * x_t @ x_t.T 
    
    return M

@jit(nopython=True)
def noisy_power_iter(M, Q):
    Q = M @ Q
    Q, _ = np.linalg.qr(Q)
    
    return Q

def oja_PCA(distance_list, init_Q, zeta):
    Q = init_Q
    for l in range(T//T_save):
        
        for t in range(T_save):
            x_t = SVD_base_._generate_xt()
            Q = oja_iter(Q, x_t, zeta, t)
            
        U = SVD_base_._get_orthosp(copy=True)
        dist = get_dist(U, Q)
        distance_list.append(dist)
        
    return distance_list

def noisy_power_PCA(distance_list, init_Q, B):
    Q = init_Q
    for l in range(T//B):
        
        M = np.zeros((p,p))
        for t in range(B):
            x_t = SVD_base_._generate_xt()
            M = noisy_power_stack(M, B, x_t)
        Q = noisy_power_iter(M, Q)
        
        U = SVD_base_._get_orthosp(copy=True)
        dist = get_dist(U, Q)
        distance_list.append(dist)
        
    return distance_list
          
#### Set Algorithm #######################################################

if algo == 'oja':
    run_PCA, lparam_str = oja_PCA, 'zeta'
    learning_param_list = np.unique(np.linspace(zeta_list[lower_idx], zeta_list[upper_idx] , 50))
    
elif algo == 'noisy':
    run_PCA, lparam_str = noisy_power_PCA, 'B'
    learning_param_list = np.unique(np.linspace(B_list[lower_idx], B_list[upper_idx] , 50,dtype=np.int32))
    
else:
    raise NotImplementedError

#### Main ###########################################################
print_status()    
datadict_list = []
start, end = 0.0, 0.0
for loopno in range(loop):    
    datadict = {}
    for lparam_idx, lparam in enumerate(learning_param_list):
        # Print current info
        description = 'Samples: ({:02d} / {:02d}) | {}: ({:02d} / {:02d}) | last-loop time: {:d}s '.format(\
            loopno+1,loop,lparam_str,lparam_idx+1,len(learning_param_list),int(end-start))
        print(description)
        
        start = time.time()
        
        # Initialization
        SVD_base_ = SVD_base(p,k,sigma,delta,gamma,rotate_type=rotate_type)
        U = SVD_base_._get_orthosp(copy=False)
        M = np.random.randn(p, k)
        Q, _ = np.linalg.qr(M)
        
        # Define distance array
        dist_list  = [get_dist(U, Q)]
        
        # main
        datadict[lparam] = run_PCA(dist_list, Q, lparam)
        
        end = time.time()
        
    datadict_list.append(datadict)

#### SAVE ###############################################################
import os, datetime, pickle

savepath = './finetuned/{}/{}'.format(algo,savelabel)
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('New directory was created.')

now = datetime.datetime.now()
formattedDate = now.strftime("%f")
savedir = savepath+'/{}.pickle'.format(formattedDate)

hyperdict = {'T':T, 'T_save':T_save, 'loop':loop, 'k':k, 'p':p, 'sigma':sigma,\
            'gamma':gamma, 'delta':delta,\
            'algo':algo, 'lp_list':learning_param_list, 'label':savelabel}
with open(savedir, 'wb') as f:
    pickle.dump(hyperdict, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(datadict_list, f, pickle.HIGHEST_PROTOCOL)
