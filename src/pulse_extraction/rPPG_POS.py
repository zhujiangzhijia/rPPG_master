"""
POS_WANG The Plane Orthogonal to Skin-Tone (POS) Method from:
　Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG.


"""
# coding: utf-8
import numpy as np
import math
from .. import preprocessing


def POSMethod(rgb_components, WinSec=1.60, LPF=0.7, HPF=2.5, fs=30):
    """
    POS method
    WinSec :was a 32 frame window with 20 fps camera
    """

    # preprocessing
    ##########################
    #  未実装
    ##########################
    # for i in range(3):
    #     rgb_components[:,i] = preprocessing.ButterFilter(rgb_components[:,i], LPF, HPF, fs)

    
    # 初期化
    N = rgb_components.shape[0]
    H = np.zeros(N)
    l = math.ceil(WinSec*fs)

    # loop from first to last frame
    for t in range(N-l+1):
        # spatical averagining
        C = rgb_components[t:t+l, :]
        # temporal normalization
        Cn = C/np.average(C, axis=0)
        # projection (orthogonal to 1)
        S = np.dot(Cn, np.array([[0,1,-1],[-2,1,1]]).T)
        # alpha tuning
        P = np.dot(S, np.array([[1, np.std(S[:,0]) / np.std(S[:,1])]]).T)
        # overlap-adding
        H[t:t+l] = H[t:t+l] + (np.ravel(P)-np.mean(P))/np.std(P)

    filtered_sig = preprocessing.ButterFilter(H, LPF, HPF, fs)
    return filtered_sig


