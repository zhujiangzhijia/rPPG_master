"""
peak to peak detector
"""
# coding: utf-8
import numpy as np

def IppgPeakDetection(ippg,fs=30):
    pass


def IppgPeakCorrection(ippg,fs=30):
    pass

def SplineInterpolate(ippg, fs=30):
    tmStamp = np.cumsum(rri)
    tmStamp -= tmStamp[0]
    
    # 3次のスプライン補間
    rri_spline = interpolate.interp1d(tmStamp, rri, 'cubic')
    t_interpol = np.arange(tmStamp[0], tmStamp[-1], 1000./sample_rate)
    return ippg_spline

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