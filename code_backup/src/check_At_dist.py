import math, random, argparse
from tqdm import tqdm
import numpy as np
from scipy import linalg, stats

from numba import jit
import warnings
warnings.filterwarnings("ignore")

from generate_At import SVD_base

dist_list = []

SVD_base_ = SVD_base(100,5,0.150,1.00,0.20)
A0 = SVD_base_._get_At(copy=True)

for _ in range(48000):
    _  = SVD_base_._generate_xt()
    A1 = SVD_base_._get_At(copy=True)
    dist = np.linalg.norm(A1 @ A1.T - A0 @ A0.T, ord=2)
    dist_list.append(dist)
    A0 = A1
    
np.save('./data/dists',np.asarray(dist_list))
