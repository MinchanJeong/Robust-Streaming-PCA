import subprocess, os, argparse, datetime
import numpy as np

algo = 'oja'
sigma = 1.0
delta = 1.0
loop = 30
label = 'defaultlabel'

parser = argparse.ArgumentParser(description='Mainly for online optimization.')
parser.add_argument('-algo', type=str, nargs='?', default = algo, help='noisy or oja(str)')
parser.add_argument('-sigma', type=float, nargs='?', default = sigma, help='sigma(float)')
parser.add_argument('-delta', type=float, nargs='?', default = delta, help='delta(float)')
parser.add_argument('-loop', type=int, nargs='?', default = loop, help='loop(int)')
parser.add_argument('-label', type=str, nargs='?', default = label, help='pickle tail(str)')
parser.add_argument('-earlystop', action='store_true', help='set earlystopping')
args = parser.parse_args()

algo = args.algo
sigma  = args.sigma 
delta  = args.delta 
loop   = args.loop 
label = args.label

if algo == 'oja':
    gamma_list = np.asarray([0.00, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]) * delta
    
elif algo == 'noisy':
    gamma_list = np.asarray([0.00, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]) * delta
    
else:
    raise NotImplementedError
    
base_arg = ' -algo {} -sigma {} -delta {} -loop {} -label {} '.format(algo, sigma, delta, loop, label)
base_arg += '-earlystop ' if args.earlystop else '' 
for gamma in gamma_list:
    os.system('python3 ./main.py'+ base_arg + '-gamma {}'.format(gamma))

now = datetime.datetime.now()
formattedDate = now.strftime("%Y%m%d_%H%M%S")
savepath = './data/{}'.format(label)
with open(savepath+'/'+'info.txt','a') as f:
    f.write(base_arg+' [{:}]\n'.format(formattedDate))
