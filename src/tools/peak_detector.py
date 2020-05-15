"""
peak to peak detector
"""
# coding: utf-8
import numpy as np
from scipy import interpolate
from .. import preprocessing
import matplotlib.pyplot as plt

def RppgPeakDetection(ppg, ts=None, fs=30, fr=100):
    """
    rPPG peak検出
    peak時間を返り値とする
    """
    # Resampling
    t_interpol, resamp_rppg = resampling(ppg, ts, fs, fr)
    # Moving Average 未実装...合ってもいいかも
    #smooth_rppg = preprocessing.MovingAve(H, num=30)
    # band pass filter
    resamp_rppg = preprocessing.ButterFilter(resamp_rppg, 0.7, 2.5, fs=fr)
    # Derivative
    ppg_dot = np.gradient(resamp_rppg, 1/fr)
    # Simple binary filter
    binary = np.copy(ppg_dot)
    binary[binary >= 0] = 1
    binary[binary < 0] = 0
    # import matplotlib.pyplot as plt
    # fig,axes = plt.subplots(2,1,sharex=True)
    # axes[0].plot(resamp_rppg)

    # axes[1].plot(binary)
    
    # plt.show()
    # zero cross
    binary_diff = np.diff(binary)
    setlists = np.where(binary_diff > 0)[0]
    # peak detection
    peak_indexes = np.array([[]])
    for i in range(len(setlists)-1):
        onset = setlists[i]
        offset = setlists[i+1]
        peak_index = onset + np.argmax(resamp_rppg[onset: offset])
        peak_indexes = np.append(peak_indexes, peak_index)
        
    return t_interpol[peak_indexes.astype(np.int64)]

def RppgPeakCorrection(RRIpeaks):
    """
    RPPGの外れ値除去
    """
    i = 2
    while RRIpeaks.shape[0] > i: 
        rri = RRIpeaks[i] - RRIpeaks[i-1]
        pre_rri = RRIpeaks[i-1] - RRIpeaks[i-2]
        if abs(rri-pre_rri) >= 0.15*pre_rri:
            RRIpeaks = np.delete(RRIpeaks, i)
        i = i + 1
    return RRIpeaks

def resampling(rppg, ts=None, fs=30, fr=100):
    """
    リサンプリング
    3次のスプライン補間
    """
    # Resampling
    if ts is None:
        ts = np.arange(0, len(rppg)/fs, 1/fs)

    rppg_interpol = interpolate.interp1d(ts, rppg, "cubic")
    t_interpol = np.arange(ts[0], ts[-1], 1./fr)
    resamp_rppg = rppg_interpol(t_interpol)
    return t_interpol, resamp_rppg


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = np.loadtxt("./result/rppg_2020-04-30_motion_talking.csv",delimiter=",")
    # smooth
    sm_size = int(0.75 * sampling_rate)
    filtered, _ = st.smoother(signal=aux,
                              kernel='boxzen',
                              size=sm_size,
                              mirror=True)
    psi_dot_np = np.gradient(data, 1/30)
    psi_2dot_np = np.gradient(psi_dot_np, 1/30)
    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(data)
    axes[1].plot(psi_dot_np**2)
    axes[2].plot(psi_2dot_np**2)
    plt.show()