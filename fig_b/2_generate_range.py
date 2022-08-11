import math, pickle, glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
from scipy.stats import linregress

for workspace in ['exp1b_finetunebase_noisy', 'exp1b_finetunebase_oja']:

    label_dir = './data/{}/'.format(workspace)
    wild_dir  = './data/{}/*.pickle'.format(workspace)

    list_for_lpwise_dicts = []

    count = 0

    print(glob.glob(wild_dir))
    for data_path in glob.glob(wild_dir):
        with open(data_path, 'rb') as f:
            hyperdict  = pickle.load(f)
            lpwise_dict = pickle.load(f)
            list_for_lpwise_dicts.append((hyperdict, lpwise_dict))
            count += 1 

    algo = hyperdict['algo']
    T, T_save, loop, k, p = hyperdict['T'], hyperdict['T_save'] ,hyperdict['loop'], hyperdict['k'], hyperdict['p']
    sigma, delta = hyperdict['sigma'], hyperdict['delta']
    learning_param_list = np.asarray(hyperdict['lp_list'])
    
    print(f'\nAlgorithm: {algo}')
    
    print('{} data were loaded.'.format(count))

    #Gamma selection
    allgamma = [x[0]['gamma'] for x in list_for_lpwise_dicts]

    print([x[0]['gamma'] for x in list_for_lpwise_dicts])

    if algo == 'noisy':
        noisydict = list_for_lpwise_dicts
        noisy_param_list   = learning_param_list
    elif algo == 'oja':
        ojadict = list_for_lpwise_dicts
        oja_param_list   = learning_param_list
    else:
        raise NotImplementedError

    print('Hyper Dictionary Sample: ')
    for k,v in hyperdict.items():
        print('{}: {}'.format(k,v))

    targetexist = True
    #target = np.asarray([0.00625, 0.0085, 0.0095]) * delta

    target = allgamma

    dictlp = {}
    for c, (hyperdict, datadict_list) in enumerate(list_for_lpwise_dicts):

        gamma = hyperdict['gamma']

        if not np.any(np.isclose(gamma, target)) or np.isclose(gamma, 0.0):
            continue

        dist_array = np.zeros((loop,len(learning_param_list)))

        dist_vec_list = []
        for datadict in datadict_list:
            dist_vec = np.zeros((1,len(learning_param_list)))
            for jdx,lp in enumerate(learning_param_list):
                dist_vec[0,jdx] = np.average(datadict[lp][-15:])

            dist_vec_list.append(dist_vec)

        dist_array = np.vstack(dist_vec_list) 

        avg = np.average(dist_array,axis=0)
        std = np.std(dist_array,axis=0)

        arg = np.argmin(avg)

        if 0 < arg < len(learning_param_list)-1:
            dictlp[gamma] = (max(arg-2,0),min(arg+2,21))


    print(dictlp)

    with open(algo+'_lpidxdict.pickle', 'wb') as f:
        pickle.dump(dictlp, f, pickle.HIGHEST_PROTOCOL)