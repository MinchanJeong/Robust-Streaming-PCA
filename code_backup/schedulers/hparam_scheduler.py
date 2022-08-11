import subprocess, os, argparse, datetime
import numpy as np

##################################################################################################\
sigma= 0.01
algo = 'noisy'
p    = 100

parser = argparse.ArgumentParser(description='Mainly for online optimization.')
parser.add_argument('-algo', type=str, nargs='?', help='noisy or oja(str)')
parser.add_argument('-p', type=int, nargs='?', help='p(int)')
args = parser.parse_args()

algo = args.algo
p  = args.p

if not (algo and p):
    print('Please give both -algo and -p arguments!!')
    raise ValueError

##################################################################################################    
base_arg = ' -algo {} -space {} -p {} -sigma {} -loop 20 '.format(algo, 'npm100_k_frob_'+algo,p,sigma) #원래는 20개!

k_list     = np.arange(5,p-9,5)
delta_list = [0.2]
gamma_list = [1.0e-5]

for i,delta in enumerate(delta_list):
    for j,gamma in enumerate(gamma_list):
        for l, k in enumerate(k_list):
            #label = '{}_{}{}'.format(p,i,j)
            label = '{}_{}'.format(p,k)
            arg = base_arg + '-delta {} -gamma {} -k {} -label {}'.format(delta, gamma, k, label)
            
            os.system('python3 ./main.py'+ arg)