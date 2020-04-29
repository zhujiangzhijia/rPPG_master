"""
評価指標を算出する
"""
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def SNR(ppg,HR, fs=30):
    """
    SNRを算出する
    SNR 大：脈拍の影響大，
    SNR 小：ノイズの影響大
    """
    HR_F = HR/60

    NyquistF = fs/2;
    FResBPM = 0.1 # パワースペクトルの解像度（bpm）
    N = (60*2*NyquistF)/FResBPM

    freq, power = signal.periodogram(ppg, fs, nfft = N, detrend="constant",
                                     scaling="spectrum", window="hamming")

    # 0.2Hz帯
    GTMask1 = (freq >= HR_F-0.1) & (freq <= HR_F+0.1)
    GTMask2 = (freq >= HR_F*2-0.2) & (freq <= HR_F*2+0.2)

    SPower = np.sum(power[GTMask1 | GTMask2])
    FMask2 = (freq >= 0.5)&(freq <= 4)
    AllPower = np.sum(Pxx(FMask2))

    # ここ怪しい
    SNR = 10*np.log10(SPower/(AllPower-SPower))
    return SNR
    pass
