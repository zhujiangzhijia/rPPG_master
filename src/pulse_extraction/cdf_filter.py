"""
rPPGの前処理
Color-distortion filtering for remote photoplethysmography
"""
# coding: utf-8
import numpy as np
import math
from scipy.fftpack import fft, ifft,fftfreq
import matplotlib.pyplot as plt


def cdf_filter(C_rgb, LPF, HPF, fs):
    """
    Color-distortion filtering for remote photoplethysmography. 
    """
    L = C_rgb.shape[0]
    # temporal normalization
    Cn = C_rgb/np.average(C_rgb, axis=0) -1 
    # FFT transform
    FF = fft(Cn,n=L,axis=0)
    freq = fftfreq(n=L, d=1/fs)
    # Characteristic transformation
    H = np.dot(FF, (np.array([[-1, 2, -1]])/math.sqrt(6)).T)
    # Energy measurement
    W = (H * np.conj(H)) / np.sum(FF*np.conj(FF), 1).reshape(-1, 1)
    # band limitation
    FMask = (freq >= LPF)&(freq <= HPF)
    FMask = FMask + FMask[::-1]
    W = W*FMask.reshape(-1, 1)
    # Weighting
    Ff = np.multiply(FF, (np.tile(W, [1, 3])))
    # temporal de-normalization
    C = np.array([(np.average(C_rgb, axis=0)),]*(L)) * np.abs(np.fft.ifft(Ff, axis=0)+1)
    return C
