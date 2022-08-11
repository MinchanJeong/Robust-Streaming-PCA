import math, random, argparse, time, sys
import numpy as np

from numba import jit
import warnings
warnings.filterwarnings("ignore")

from data_stream import stream

T_oja = 100

#### Hyperparameters #####################################################
clip = False
loop = 5
testwindow = 500

k = 3

B_list = np.unique(np.asarray(np.power(10,np.linspace(start = 0, stop = 3.2, num = 400)),dtype=np.int))#400
zeta_list = np.power(10,np.linspace(start = 2.5, stop = -3.5, num = 200))#200 2.5 ~ -1.5
    
#### Argparse ############################################################
parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('-algo', type=str, nargs='?', default = 'oja', help='noisy or oja(str)')
parser.add_argument('-loop', type=int, nargs='?', default = loop, help='loop(int)')

parser.add_argument('-k', type=int, nargs='?', default = k, help='latentdim')
parser.add_argument('-testwindow', type=int, nargs='?', default = testwindow, help='latentdim')

parser.add_argument('-clip', action='store_true', default = False)

args = parser.parse_args()

algo = args.algo
k, loop, testwindow = args.k, args.loop, args.testwindow
clip = args.clip
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

def oja_PCA(init_Q, zeta, T):
    Q = init_Q
        
    for t in range(T):
        x_t = stream_._generate_xt()
        Q = oja_iter(Q, x_t, zeta, t)
            
    U = stream_._get_testsp(copy=True)
    Q, _ = np.linalg.qr(Q)
    dist = get_dist(U, Q)
        
    return dist

def noisy_power_PCA(init_Q, B, T):
    Q = init_Q
    for l in range(T//B):
        
        M = np.zeros((p,p))
        for t in range(B):
            x_t = stream_._generate_xt()
            M = noisy_power_stack(M, B, x_t)
        Q = noisy_power_iter(M, Q)
        
    U = stream_._get_testsp(copy=True)
    dist = get_dist(U, Q)
        
    return dist
        
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
    stream_ = stream(k, testwindow, clip = clip)
    for lparam_idx, lparam in enumerate(learning_param_list):
        # Print current info
        description = 'Samples: ({:02d} / {:02d}) | {}: ({:02d} / {:02d}) | last-loop time: {:d}s '.format(\
            loopno+1,loop,lparam_str,lparam_idx+1,len(learning_param_list),int(end-start))
        print(description)
        
        # Reinitialize the datastream
        adjust = 0 if algo=='oja' else lparam.astype(int)
        T, p = stream_._reset(adjust)
        
        start = time.time()
        
        # Initialization        
        M = np.random.randn(p, k)
        Q, _ = np.linalg.qr(M)
        
        # main
        datadict[lparam] = run_PCA(Q, lparam, T)
    
        end = time.time()
        
    datadict_list.append(datadict)

#### SAVE ###############################################################
import os, datetime, pickle

savepath = f'./result/sp500_{k}'
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('New directory was created.')

now = datetime.datetime.now()
formattedDate = now.strftime("%f")
#savedir = savepath+'/{}.pickle'.format(formattedDate)
if clip == False:
    savedir = savepath+'/algo_{}_wdw_{}_loop_{}_{}.pickle'.format(algo,testwindow, loop, formattedDate[-3:])
else:
    savedir = savepath+'/algo_{}clip_wdw_{}_loop_{}_{}.pickle'.format(algo,testwindow, loop, formattedDate[-3:])

hyperdict = {'loop':loop, 'k':k,'testwindow':testwindow, 'algo':algo, 'lp_list':learning_param_list, 'clip':clip}
with open(savedir, 'wb') as f:
    pickle.dump(hyperdict, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(datadict_list, f, pickle.HIGHEST_PROTOCOL)
