"""
SoftSig
参考URL:https://ushitora.net/archives/1016
"""
# coding: utf-8
import numpy as np
import math
from .. import preprocessing
import matplotlib.pyplot as plt
from scipy import fftpack

def SoftsigMethod(rgb_components, WinSec=3.2, LPF=0.6, HPF=3.0, fs=None, filter=True):
    """
    Softsig method
    WinSec :was a 32 frame window with 20 fps camera
    (i) L = 32 (1.6 s)
    (ii) L = 64 (3.2 s) <-recommended
    (iii) L = 128 (6.4 s) 
    (iv) L = 256 (12.8 s) 
    (v) L = 512 (25.6 s) 
    """

    # 初期化
    N = rgb_components.shape[0]
    H = np.zeros(N)
    l = math.ceil(WinSec*fs)

    # loop from first to last frame
    for t in range(N-l+1):
        print("{} | {} %".format(t,t/(N-l+1)*100))
        # temporal normalization
        C = rgb_components[t:t+l, :]
        Cn = C / np.average(C, axis=0) - 1
        # softsig
        w = softsig_selection(Cn,fs)
        print(w)
        P = np.dot(Cn,w.reshape(-1,1))
        # overlap-adding
        H[t:t+l] = H[t:t+l] + (np.ravel(P)-np.mean(P))/np.std(P)

    if filter:
        # Buffer filter
        H = preprocessing.ButterFilter(H, LPF, HPF, fs)
    return H

# Simpler Proposed Method (no physiological conditions)
def softsig_selection(Cn,fs,step=40):
    # R,G,B ... evenly spaced time intervals
    # 格子点の作成
    theta_range = np.linspace(0, np.pi/2, step) # θ_1は[0,π/2]の値をとる
    theta,phi = np.meshgrid(theta_range, theta_range)
    vR = np.cos(theta)*np.sin(phi)# xの極座標表示
    vG = np.sin(theta)*np.sin(phi)# yの極座標表示
    vB = np.cos(phi) # zの極座標表示
    points = np.stack((vR,vG,vB),axis=-1).reshape(-1,3)
    
    # G>R and G>B
    rgb_range = ((points[:,1]>points[:,2]) & (points[:,1]>points[:,0]))
    points = points[rgb_range]

    snr = np.array([[[]]])
    # selection
    for point in points:
        # projection 
        P = np.dot(Cn,point.reshape(-1,1))
        # evaluate SNR
        snr = np.append(snr, softsig_fft(P, fs))
    return points[np.argmax(snr),:]


def softsig_fft(ppg,fs,type="max"):
    """
    type = max or mean
    if mean
        SN(1,k) = M / (sum(spec(:,2))-M) %<-SoftSig
    if max
        SN(1,k) = Max / (mean(power)-Max) %<-SoftSig
    """
    L = len(ppg)
    
    # FFT
    fft_amp = fftpack.fft(ppg)  # 周波数領域のAmplitude
    fft_fq = fftpack.fftfreq(n=L, d=1.0/fs)  # Amplitudeに対応する周波数
    
    # 正の領域のみ抽出
    fft_amp = fft_amp[0: int(len(fft_amp)/2)]
    fft_fq = fft_fq[0: int(len(fft_fq)/2)]
    fft_amp = abs(fft_amp)  # 複素数→デシベル変換

    FMask2 = (fft_fq >= 0.66)&(fft_fq<= 4)   
    if type == "max":
        S_max = np.max(fft_amp[FMask2])
        S_sum = np.sum(fft_amp[FMask2])
        snr_f = S_max/(S_sum-S_max)
        

    elif type == "mean":
        # Define estimated HF
        # 間違っている
        HR_F = freq[np.argmax(fft_amp[FMask2])]
        print(60*HR_F)
        # 0.2Hz帯
        GTMask1 = (freq >= HR_F-0.1) & (freq <= HR_F+0.1)
        GTMask2 = (freq >= HR_F*2-0.2) & (freq <= HR_F*2+0.2)
        SPower = np.sum(amplitude[GTMask1 | GTMask2])
        AllPower = np.sum(amplitude[FMask2])
        snr_f = SPower/(AllPower-SPower)

    return snr_f


if __name__ == "__main__":
    data=np.loadtxt(r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\effect_of_FPS\100Hz\rgb_signal_100Hz.csv",delimiter=",")[800:,:]

    # softsig_fft(data[:60000,1],fs=100,type="max")

    plt.plot(SoftsigMethod(data,fs=100))
    plt.show()