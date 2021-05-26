# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath

from numpy.core.numeric import identity
from numpy.ma.extras import cov
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')  

import wasserstein as ws
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

dim = 3
id_ = 0
particle_count = 800
num_exps = 10
cov_folder = 'cov/{}_pc_{}'.format(id_, particle_count)
save_path = cov_folder + '/zero.npy'
gap = 4
ev_time = 400


def tf_sampler(mean, cov, size):
    samples = np.random.multivariate_normal(mean, cov, size)
    return tf.convert_to_tensor(samples, dtype=tf.float32)

mean = np.zeros(dim)
id3 = np.identity(dim)
w_s = np.zeros(num_exps)
zero = np.zeros(int(ev_time/gap))

df = pd.read_csv(cov_folder + '/')

for j in range(0, ev_time, gap):
    print('assimilation step #{}'.format(j))
    for i in range(num_exps):
        print('working on iteration #{}'.format(i), end='\r')
        samples_1 = tf_sampler(mean, cov, particle_count)
        samples_2 = tf_sampler(mean, cov, particle_count)
        w_s[i] = np.sqrt(ws.sinkhorn_div_tf(samples_1, samples_2, epsilon=0.01, num_iters=50, p=2).numpy())
    zero[j] = np.mean(w_s) 