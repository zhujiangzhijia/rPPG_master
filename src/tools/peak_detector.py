"""
peak to peak detector
"""
# coding: utf-8
import numpy as np
from scipy import interpolate
from .. import preprocessing
import matplotlib.pyplot as plt

def RppgPeakDetection(ppg,fs,fr=100, show=False, filter=False, col=0.7):
    """
    rPPG peak検出
    peak時間を返り値とする
    """
    # Moving Average
    if filter==True:
        #ppg = preprocessing.MovingAve(ppg, num=3)
        ppg =  preprocessing.ButterFilter(ppg, 0.7, 2.5, fs)
    
    # Resampling
    t_interpol, resamp_rppg = resampling(ppg, fs, fr)
    
    # 1st Derivative
    ppg_dot = np.gradient(resamp_rppg, 1/fr)

    # Simple binary filter
    binary = np.copy(ppg_dot)
    binary[binary >= 0] = 1
    binary[binary < 0] = 0
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

    # peak correction 
    rpeaks = t_interpol[peak_indexes.astype(np.int64)]
    rppg_amp = resamp_rppg[peak_indexes.astype(np.int64)]
    rpeaks = rpeaks[(rppg_amp>np.mean(rppg_amp)*col)]

    if show == True:
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(t_interpol, resamp_rppg)
        axes[0].set_title("Resample signal")
        for rpeak in rpeaks:
            axes[0].axvline(rpeak)
        axes[1].set_title("RRI signal")
        axes[1].plot(rpeaks[1:],rpeaks[1:]-rpeaks[:-1])
        plt.show()

    return rpeaks*1000

def RppgPeakCorrection(RRIpeaks, col=0.80):
    """
    RPPGの外れ値除去
    """
    i = 2
    while RRIpeaks.shape[0] > i: 
        rri = RRIpeaks[i] - RRIpeaks[i-1]
        pre_rri = RRIpeaks[i-1] - RRIpeaks[i-2]
        if rri/pre_rri <= col:
            RRIpeaks = np.delete(RRIpeaks, i)
        i = i + 1
    return RRIpeaks

def resampling(rppg, fs, fr):
    """
    リサンプリング
    3次のスプライン補間
    """
    ts = np.arange(0, len(rppg)/fs, 1./fs)
    rppg_interpol = interpolate.interp1d(ts, rppg, "cubic")
    t_interpol = np.arange(ts[0], ts[-1], 1./fr)
    resamp_rppg = rppg_interpol(t_interpol)
    return t_interpol, resamp_rppg

def linear_resampling(rppg, ts, fr=100):
    """
    リサンプリング
    3次のスプライン補間
    """
    #ts = np.arange(0, len(rppg)/fs, 1./fs)[:int(len(rppg))]
    rppg_interpol = interpolate.interp1d(ts, rppg, "linear")
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