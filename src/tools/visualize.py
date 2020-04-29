"""
可視化用
"""
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def plot_snr(ppg, fs=30):
    """
    SNRを算出する
    SNR が大きいと脈拍の影響が大きく，
    SNRが小さいとノイズの影響が大きい
    """
    NyquistF = fs/2;
    FResBPM = 0.1 # パワースペクトルの解像度（bpm）
    N = (60*2*NyquistF)/FResBPM

    freq, power = signal.periodogram(ppg, fs, nfft = N, detrend="constant",
                                     scaling="spectrum", window="hamming")
    plt.figure()
    plt.plot(freq*60, power)
    plt.xlabel("Frequency [bpm]")
    plt.ylabel("Normalized Amplitude [-]")
    plt.xlim(0,250)


def plot_spectrograms():
    pass