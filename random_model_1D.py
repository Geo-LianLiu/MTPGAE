# ======================================================================== #
# Copyright 2019 Lian Liu. All Rights Reserved.                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #
#                       @Author  :   Lian Liu                              #
#                       @e-mail  :   lianliu1017@126.com                   #
# ======================================================================== #

# Import modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def random_model(nz, filterfactor=9, iters=50, bounds=None):
    # form filter
    filter = np.array([1, filterfactor, 1], dtype=np.float64)
    
    # normalize
    filter = filter / filter.sum()

    if bounds is None:
        bounds = [0, 4]

    # random model
    seed = np.random.randint(1e5)
    np.random.seed(seed)
    m0 = np.random.rand(nz)

    mk = m0
    for _ in range(iters):
        mk = ndi.convolve(mk, filter)

    # scale the model
    mk = (mk - mk.min()) / (mk.max() - mk.min())

    a = np.random.uniform(bounds[0],bounds[1])
    b = np.random.uniform(bounds[0],bounds[1])
    if a >= b:
        mk = mk * (a - b) + b
    else:
        mk = mk * (b - a) + a

    return mk

def plot_model(rho, zn):
    rho = np.concatenate([rho, rho[-2:-1]], axis=0)
    plt.figure(figsize = (4, 3))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = '10'
    plt.rcParams['ytick.major.size'] = '10'
    plt.rcParams['xtick.minor.size'] = '5'
    plt.rcParams['ytick.minor.size'] = '5'
    plt.step(zn, rho, linewidth = 1.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim((1e0, 1e4))
    plt.ylim((1e-1, 1e5))
    plt.xlabel('Depth (m)', fontsize = 12)
    plt.ylabel('Resistivity (ohm.m)', fontsize = 12)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    # plt.legend(['Real'], loc = 'best', fontsize = 11)
    plt.savefig("randommodel.pdf")
    

if __name__=='__main__':
    dz = 20. + 10.**(0.115*np.linspace(1, 30, 30))
    dz = np.concatenate([[0.], dz], axis=0)
    zn = np.concatenate([[0.], np.cumsum(dz)], axis=0)
    zn = np.concatenate([[0.], np.cumsum(dz)], axis=0)
    nz = len(dz)
    
    model = random_model(nz, filterfactor=9, iters=30, bounds=[-1, 5])
    rho = 10.**model
    plot_model(rho, zn)
