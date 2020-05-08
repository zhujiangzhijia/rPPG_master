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
