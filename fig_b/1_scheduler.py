import subprocess, os, argparse, datetime
import numpy as np

algo = 'oja'
sigma = 0.15
delta = 1.0
loop = 3
workspace = 'defaultspace'

parser = argparse.ArgumentParser(description='Mainly for online optimization.')
parser.add_argument('-algo', type=str, nargs='?', default = algo, help='noisy or oja(str)')
parser.add_argument('-sigma', type=float, nargs='?', default = sigma, help='sigma(float)')
parser.add_argument('-delta', type=float, nargs='?', default = delta, help='delta(float)')
parser.add_argument('-loop', type=int, nargs='?', default = loop, help='loop(int)')
parser.add_argument('-space', type=str, nargs='?', default = workspace, help='workspace(str)')
args = parser.parse_args()

algo = args.algo
sigma  = args.sigma 
delta  = args.delta 
loop   = args.loop 
workspace = args.space

if algo == 'oja' or algo == 'noisy':
    #gamma_list = np.asarray([])# * delta
    gamma_list = np.asarray([0.00001,0.00002,0.00003,0.00004,0.00005,\
                             0.00006,0.00007,0.00008,0.00009,0.0001,1.05e-5,\
                             1.10e-5,1.15e-5,1.21e-5,1.27e-5,1.33e-5,1.40e-5,\
                             1.48e-5,1.56e-5,1.65e-5,1.76e-5,1.87e-5,2.18e-5,\
                             2.35e-5,2.53e-5,2.74e-5,3.26e-5,3.58e-5,4.42e-5])
    gamma_list = np.asarray([0.00001,\
                             0.00006,\
                             1.10e-5,\
                             1.48e-5,\
                             2.35e-5])    

else:
    raise NotImplementedError
    
base_arg = ' -algo {} -sigma {} -delta {} -loop {} -space {} '.format(algo, sigma, delta, loop, workspace)
#base_arg += '-p 2 '
for gamma in gamma_list:
    os.system('python3 ./main.py'+ base_arg + '-gamma {}'.format(gamma))

now = datetime.datetime.now()
formattedDate = now.strftime("%Y%m%d_%H%M%S")
savepath = './data/{}'.format(workspace)
with open(savepath+'/'+'info.txt','a') as f:
    f.write(base_arg+' [{:}]\n'.format(formattedDate))
