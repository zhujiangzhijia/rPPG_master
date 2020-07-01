"""
評価指標を算出する
"""
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def CalcSNR(ppg, HR_F=None, fs=30, nfft=512):
    """
    CHROM参照
    SNRを算出する
    SNR が大きいと脈拍の影響が大きく，
    SNRが小さいとノイズの影響が大きい
    """
    freq, power = signal.welch(ppg, fs, nfft=nfft, detrend="constant",
                                     scaling="spectrum", window="hamming")
    # peak hr
    if HR_F is None:
        HR_F = freq[np.argmax(power)]

    # 0.2Hz帯
    GTMask1 = (freq >= HR_F-0.1) & (freq <= HR_F+0.1)
    GTMask2 = (freq >= HR_F*2-0.2) & (freq <= HR_F*2+0.2)
    SPower = np.sum(power[GTMask1 | GTMask2])
    FMask2 = (freq >= 0.5)&(freq <= 4)
    AllPower = np.sum(power[FMask2])
    SNR = 10*np.log10((SPower)**2/(AllPower-SPower)**2)
    return HR_F, SNR

def CalcFreqHR(ppg, fs=30, nfft=512):
    """
    Calculate Frequency domain heart rate
    using DFT,
    return HR[bpm]
    """
    # FFT PSD
    f, t, Sxx = signal.spectrogram(ppg, fs, nperseg=nfft,
                                   noverlap=nfft/2, scaling="spectrum")
    # Calc HR
    HR_F = np.array([[]])
    for i in range(len(t)):
        Sxx_t = Sxx[:, i]
        HR_F_t = 60*f[np.argmax(Sxx_t)]
        HR_F = np.append(HR_F, HR_F_t)
    return t, HR_F
    
def CalcTimeHR(rpeaks, rri, segment=17.06, overlap=None):
    """
    Time domain Heart rate
    """
    if overlap is None:
        overlap = segment/2 
    starts = np.arange(0, rpeaks[-1]-overlap, overlap)
    HR_T = np.array([[]])
    for start in starts:
        end = start + segment
        item_rri = rri[(rpeaks >= start) & (rpeaks < end)]
        ave_hr = 60/np.average(item_rri)
        HR_T = np.append(HR_T, ave_hr)
    ts = starts + overlap
    return ts, HR_T 