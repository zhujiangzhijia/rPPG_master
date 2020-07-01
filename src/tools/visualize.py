"""
可視化用
"""
# coding: utf-8
import numpy as np
from scipy import signal,interpolate
from .. import preprocessing 
from . import peak_detector
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
    print("Result index: MAE={} RMSE={}".format(mae,rmse))
    plt.figure(figsize=(8,6))
    plt.scatter(x, y)
    plt.axhline(sygma*1.96,label="+1.96σ")
    plt.axhline(-sygma*1.96,label="-1.96σ")
    plt.axhline(np.mean(y),label="mean", color='black')
    plt.xlabel("(Estimate+Reference)/2 [ms]")
    plt.ylabel("Estimate-Reference [ms]")
    plt.title("Bland Altman Plot\nMAE={:.2f}, RMSE={:.2f}".format(mae, rmse))
    plt.legend()
    plt.show()

def plot_rri(ref_ppg,filter_rppg, nonfilter_rppg):
    filter_peaks = peak_detector.RppgPeakDetection(filter_rppg, fr=250,filter=True)
    nonfilter_peaks = peak_detector.RppgPeakDetection(nonfilter_rppg, fr=250,filter=True,show=False)
    ref_peaks = peak_detector.RppgPeakDetection(ref_ppg, fr=250, filter=True)
    ref_peaks = peak_detector.RppgPeakCorrection(ref_peaks)
    fig, axes = plt.subplots(2, 1, sharex=True)
    # # あとで消す
    # filter_peaks=filter_peaks[:int(ref_peaks.shape[0])]
    # nonfilter_peaks=nonfilter_peaks[:int(ref_peaks.shape[0])]
    axes[0].plot(ref_peaks[1:]*30, preprocessing.RRInterval(ref_peaks),label="PPG")
    axes[0].plot(filter_peaks[1:]*30, preprocessing.RRInterval(filter_peaks),label="Filter rPPG")
    axes[0].plot(nonfilter_peaks[1:]*30, preprocessing.RRInterval(nonfilter_peaks),label="Nonfilter rPPG")
    #rp,ri = preprocessing.outlier_correction(filter_peaks, threshold=0.13)
    #axes[0].plot(ref_peaks[1:]*30, ri, label="Filter rPPG")
    # axes[0].plot(ref_peaks[1:]*30, preprocessing.RRInterval(nonfilter_peaks), label="Nonfilter rPPG")
    # axes[0].scatter(ref_peaks[1:]*30, preprocessing.RRInterval(nonfilter_peaks), label="Nonfilter rPPG")
    ts = np.arange(0, len(ref_ppg)/250,1/250)
    estts = np.arange(0, len(filter_rppg)/30,1/30)
    axes[1].plot((ref_ppg-np.min(ref_ppg))/(np.max(ref_ppg)-np.min(ref_ppg)),label="PPG")
    axes[1].plot((filter_rppg-np.min(filter_rppg))/(np.max(filter_rppg)-np.min(filter_rppg)),label="POS")
    axes[1].plot((nonfilter_rppg-np.min(nonfilter_rppg))/(np.max(nonfilter_rppg)-np.min(nonfilter_rppg)), label="Green")
    plt.legend()
    plt.show()

def plot_PSD(rri_peaks, rri=None, label=None,nfft=2**10):
    sample_rate = 4
    if rri is None:
        rri = np.diff(rri_peaks)
        rri_peaks = rri_peaks[1:] - rri_peaks[1]

    # 3次のスプライン補間
    rri_spline = interpolate.interp1d(rri_peaks, rri, 'cubic')
    t_interpol = np.arange(rri_peaks[0], rri_peaks[-1], 1./sample_rate)
    rri_interpol = rri_spline(t_interpol)
    frequencies, powers  = signal.welch(x=rri_interpol, fs=sample_rate, window='hamming',
                                        detrend="constant",	nperseg=nfft,
                                        nfft=nfft, scaling='density')
    LF = np.sum(powers[(frequencies>=0.05) & (frequencies<0.15)]) * 0.10
    HF = np.sum(powers[(frequencies>0.15) & (frequencies<=0.40)]) * 0.25
    print("Result :LF={:2f}, HF={:2f}, LF/HF={:2f}".format(LF, HF, LF/HF))
    plt.plot(frequencies, powers/10**6,label=label)
    plt.axvline(x=.05, color='r')
    plt.axvline(x=0.15, color='r')
    plt.axvline(x=0.40, color='r')
    plt.xlim(0,.5)
    plt.xlabel("frequency[Hz]")
    plt.ylabel("PSD[s^2/Hz]")

    