"""
POS_WANG The Plane Orthogonal to Skin-Tone (POS) Method from:
　Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG.


"""
# coding: utf-8
import numpy as np
import math
from .. import preprocessing
from . import cdf_filter
import matplotlib.pyplot as plt

def POSMethod(rgb_components, WinSec=1.6, LPF=0.7, HPF=2.5, fs=30,filter = True):
    """
    POS method
    WinSec :was a 32 frame window with 20 fps camera
    (i) L = 32 (1.6 s), B = [3,6]
    (ii) L = 64 (3.2 s), B = [4,12] 
    (iii) L = 128 (6.4 s), B = [6,24] 
    (iv) L = 256 (12.8 s), B = [10,50] 
    (v) L = 512 (25.6 s), B = [18,100] 
    """

    # 初期化
    N = rgb_components.shape[0]
    H = np.zeros(N)
    l = math.ceil(WinSec*fs)

    # loop from first to last frame
    for t in range(N-l+1):
        # spatical averagining
        C = rgb_components[t:t+l, :]
        if filter == True:
            C = cdf_filter.cdf_filter(C, LPF, HPF, fs=fs)
            Cn = C / np.average(C, axis=0)
        else:
            # temporal normalization
            C = cdf_filter.cdf_filter(C, LPF, HPF, fs=fs, bpf=True)
            Cn = C/np.average(C, axis=0)
        # projection (orthogonal to 1)
        S = np.dot(Cn, np.array([[0,1,-1],[-2,1,1]]).T)
        # alpha tuning
        P = np.dot(S, np.array([[1, np.std(S[:,0]) / np.std(S[:,1])]]).T)
        # overlap-adding
        H[t:t+l] = H[t:t+l] + (np.ravel(P)-np.mean(P))/np.std(P)
        # fig,axes = plt.subplots(1,2,figsize=(12,4))
        # axes[0].plot(S[:,0],"y")
        # axes[0].plot(S[:,1])
        # axes[1].plot(np.ravel(P))
        # plt.savefig("test2.png")
    return H

