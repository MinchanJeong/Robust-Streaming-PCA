import math, random, argparse, time, sys
import numpy as np

from numba import jit
import warnings
warnings.filterwarnings("ignore")

sys.path.append('../src')
from generate_At import SVD_base

DEBUG = False

#### Hyperparameters #####################################################
T = 144000
T_save, T_oja = 100, 5
rotate_type = True

B_list = np.asarray([2,3,8,10,20,30,40,60,80,100,200,300,400,600,800,1000,1200,1500,1800,2000,3000,4000,6000,8000,9600])
zeta_list = 1.0 / B_list

sigma = 0.15

assert T % T_save == 0
assert T_save % T_oja == 0

if DEBUG:
    B_list = [1, 10, 100, 1000]
    zeta_list = [0.001, 0.01, 0.1, 1.0, 10.0]
    
#### Argparse ############################################################
parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('-algo', type=str, nargs='?', default = 'oja', help='noisy or oja(str)')
parser.add_argument('-loop', type=int, nargs='?', default = 10, help='loop(int)')
parser.add_argument('-p', type=int, nargs='?', default = 100, help='totaldim')
parser.add_argument('-k', type=int, nargs='?', default = 5, help='latentdim')

parser.add_argument('-sigma', type=float, nargs='?', default = sigma, help='sigma(float)')
parser.add_argument('-delta', type=float, nargs='?', default = 1.0, help='delta(float)')
parser.add_argument('-gamma', type=float, nargs='?', default = 2.0e-6, help='gamma(float)')

parser.add_argument('-space', type=str, nargs='?', default = 'defaultspace', help='workspace')
args = parser.parse_args()

algo = args.algo
p, k, loop = args.p, args.k, args.loop
sigma, delta, gamma = args.sigma, args.delta, args.gamma 

workspace = args.space

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
start, end = 0.0, 0.0

np.random.seed(0)
seeds = np.random.randint(12345679,size = loop)

for loopno in range(loop):
    np.random.seed(seeds[loopno])
    
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

savepath = './data/{}'.format(workspace)
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('New directory was created.')

now = datetime.datetime.now()
formattedDate = now.strftime("%f")
savedir = savepath+'/{}.pickle'.format(formattedDate)

hyperdict = {'T':T, 'T_save':T_save, 'loop':loop, 'k':k, 'p':p, 'sigma':sigma,\
            'gamma':gamma, 'delta':delta,\
            'algo':algo, 'lp_list':learning_param_list,'rotate_type':rotate_type}
with open(savedir, 'wb') as f:
    pickle.dump(hyperdict, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(datadict_list, f, pickle.HIGHEST_PROTOCOL)
