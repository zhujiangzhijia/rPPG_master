"""
可視化用
"""
# coding: utf-8
import numpy as np
from scipy import signal
from .. import preprocessing 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set(font_scale=8/6)

def plot_snr(ppg, hr=None, fs=30):
    """
    CHROM参照
    SNRを算出する
    SNR が大きいと脈拍の影響が大きく，
    SNRが小さいとノイズの影響が大きい
    """
    NyquistF = fs/2;
    FResBPM = 0.5 # パワースペクトルの解像度（bpm）
    N = (60*2*NyquistF)/FResBPM

    freq, power = signal.welch(ppg, fs, nfft=512, detrend="constant",
                                     scaling="spectrum", window="hamming")
    # peak hr
    HR_F = freq[np.argmax(power)]
    # 0.2Hz帯
    GTMask1 = (freq >= HR_F-0.1) & (freq <= HR_F+0.1)
    GTMask2 = (freq >= HR_F*2-0.2) & (freq <= HR_F*2+0.2)
    SPower = np.sum(power[GTMask1 | GTMask2])
    FMask2 = (freq >= 0.5)&(freq <= 4)
    AllPower = np.sum(power[FMask2])
    SNR = 10*np.log10((SPower)**2/(AllPower-SPower)**2)
    # normalize
    power_norm = (power-np.min(power))/(np.max(power)-np.min(power))
    plt.figure()
    plt.plot(freq*60, power_norm)
    plt.axvspan(60*(HR_F-0.1), 60*(HR_F+0.1), color = "coral", alpha=0.2)
    plt.axvspan(60*(2*HR_F-0.2), 60*(2*HR_F+0.2), color = "coral", alpha=0.2)
    plt.xlabel("Frequency [bpm]")
    plt.ylabel("Normalized Amplitude [-]")
    plt.xlim(0, 250)
    plt.title("freq HR: {:.2f}  SNR: {:.2f}".format(HR_F, SNR))


def plot_spectrograms(ppg, fs=30, nfft=256,title=None):
    """
    Plot spectrogram
    """
    f, t, Sxx = signal.spectrogram(ppg, fs=fs, nperseg=nfft,
                                   noverlap=nfft-1, scaling="spectrum")
    plt.pcolormesh(t, f*60, Sxx, norm=mpl.colors.LogNorm(vmin=Sxx.mean(), vmax=Sxx.max()))
    if title is not None:
        plt.title(title)
    plt.ylabel('Frequency [BPM]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, 250)
    plt.show()

def plot_BlandAltman(rppg_peak, ref_peak):
    """
    Plot Bland Altman
    """
    est = 1000*(rppg_peak[1:] -rppg_peak[:-1])
    ref = 1000*(ref_peak[1:] - ref_peak[:-1])
    x = 0.5*(est + ref)
    y = (est - ref)
    mae = np.mean(abs(y))
    rmse = np.sqrt(np.mean(y**2))
    sygma = np.std(y)
    plt.figure()
    plt.scatter(x, y)
    plt.axhline(sygma*1.96)
    plt.axhline(-sygma*1.96)
    plt.xlabel("(Estimate+Reference)/2 [ms]")
    plt.ylabel("Estimate-Reference [ms]")
    plt.title("Bland Altman Plot\nMAE={:.2f}, RMSE={:.2f}".format(mae, rmse))
    plt.show()
