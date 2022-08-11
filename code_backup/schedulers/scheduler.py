import subprocess, os, argparse, datetime
import numpy as np

algo = 'oja'
sigma = 1.0
delta = 1.0
loop = 30
label = 'defaultlabel'
workspace = 'defaultspace'

parser = argparse.ArgumentParser(description='Mainly for online optimization.')
parser.add_argument('-algo', type=str, nargs='?', default = algo, help='noisy or oja(str)')
parser.add_argument('-sigma', type=float, nargs='?', default = sigma, help='sigma(float)')
parser.add_argument('-delta', type=float, nargs='?', default = delta, help='delta(float)')
parser.add_argument('-loop', type=int, nargs='?', default = loop, help='loop(int)')
parser.add_argument('-label', type=str, nargs='?', default = label, help='pickle tail(str)')
parser.add_argument('-space', type=str, nargs='?', default = workspace, help='workspace(str)')
args = parser.parse_args()

algo = args.algo
sigma  = args.sigma 
delta  = args.delta 
loop   = args.loop 
label = args.label

if algo == 'oja':
    gamma_list = np.asarray([0.00, 0.006, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]) * delta
    
elif algo == 'noisy':
    gamma_list = np.asarray([0.00, 0.006, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]) * delta
    
else:
    raise NotImplementedError
    
base_arg = ' -algo {} -sigma {} -delta {} -loop {} -label {} -space {}'.format(algo, sigma, delta, loop, label, workspace)
for gamma in gamma_list:
    os.system('python3 ./main.py'+ base_arg + '-gamma {}'.format(gamma))

now = datetime.datetime.now()
formattedDate = now.strftime("%Y%m%d_%H%M%S")
savepath = './data/{}'.format(label)
with open(savepath+'/'+'info.txt','a') as f:
    f.write(base_arg+' [{:}]\n'.format(formattedDate))
