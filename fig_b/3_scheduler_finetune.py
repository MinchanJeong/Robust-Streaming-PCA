import subprocess, os, argparse, datetime, pickle
from tqdm import tqdm
import numpy as np

algo = 'oja'
sigma = 0.15
delta = 1.0
loop = 30
label = 'defaultlabel'

parser = argparse.ArgumentParser(description='Mainly for online optimization.')
parser.add_argument('-algo', type=str, nargs='?', default = algo, help='noisy or oja(str)')
parser.add_argument('-sigma', type=float, nargs='?', default = sigma, help='sigma(float)')
parser.add_argument('-delta', type=float, nargs='?', default = delta, help='delta(float)')
parser.add_argument('-loop', type=int, nargs='?', default = loop, help='loop(int)')
args = parser.parse_args()

algo = args.algo
sigma  = args.sigma 
delta  = args.delta 
loop   = args.loop 

with open(algo+'_lpidxdict.pickle', 'rb') as f:
    lpidxdict  = pickle.load(f)

print(lpidxdict)
gamma_n_lp_list = [(k,v[0],v[1]) for k,v in lpidxdict.items()]

    
base_arg = ' -algo {} -sigma {} -delta {} -loop {} '.format(algo, sigma, delta, loop)
for gamma,lb,ub in tqdm(gamma_n_lp_list):
    optional_arg = '-gamma {} -loweridx {} -upperidx {}'.format(gamma,lb,ub)
    os.system('python3 ./main_finetune.py'+ base_arg + optional_arg)

now = datetime.datetime.now()
formattedDate = now.strftime("%Y%m%d_%H%M%S")
savepath = f'./finetuned/{algo}'
with open(savepath+'/'+'info.txt','a') as f:
    f.write(base_arg+' [{:}]\n'.format(formattedDate))
