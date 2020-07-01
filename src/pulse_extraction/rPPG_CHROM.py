"""
CHROM_DEHAAN The Chrominance Method from:
 De Haan, G., & Jeanne, V. (2013). 
 Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886. DOI: 10.1109/TBME.2013.2266196

original:https://github.com/danmcduff/iphys-toolbox/blob/master/CHROM_DEHAAN.m

"""
# coding: utf-8
import numpy as np
import math
from .. import preprocessing
def ChromMethod(rgb_components, WinSec=1.60, LPF=0.7, HPF=2.5, fs=15):
    """
    CHROM method
    WinSec :was a 32 frame window with 20 fps camera
    """

    # Window parameters - overlap, add with 50% overlap cellは切りあげ
    WinL = math.ceil(WinSec*fs);

    # force even window size for overlap, add of hanning windowed signals
    if(WinL % 2):
        WinL += 1

    # floorは切り下げ
    NWin = math.floor((rgb_components.shape[0]-WinL/2)/(WinL/2));
    # 信号の初期化
    S = np.zeros(rgb_components.shape[0]);
    WinS = 0 # Window Start Index
    WinM = WinS+WinL/2 # Window Middle Index
    WinE = WinS+WinL #Window End Index

    for i in range(NWin):  
        # Temporal Normalize
        rgb_base = np.average(rgb_components[WinS:WinE,:], axis=0)
        rgb_norm = rgb_components[WinS:WinE,:]/rgb_base - 1
        # project to chrominance
        Xs = 3*rgb_norm[:,0] - 2*rgb_norm[:, 1] # 3Rn-2Gn
        Ys = 1.5*rgb_norm[:,0] + rgb_norm[:, 1] - 1.5*rgb_norm[:,2]# 1.5Rn+Gn-1.5Bn
        # band pass filter
        Xf = preprocessing.ButterFilter(Xs, LPF, HPF, fs)
        Yf = preprocessing.ButterFilter(Ys, LPF, HPF, fs)
        # alpha tuning
        alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf - alpha*Yf
        # overlap, add Hanning windowed signals
        wh = np.hanning(WinL)
        SWin = SWin * wh
        if i==0:
            S = SWin;
        else:
            S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL/2)] #1st half overlap
            S = np.append(S, SWin[int(WinL/2):])
    
        WinS = int(WinM)
        WinM = int(WinS + WinL/2)
        WinE = int(WinS + WinL)
    return S
