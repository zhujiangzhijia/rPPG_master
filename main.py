"""
実行ファイル
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from src.roi_detection.landmark_extractor import *
from src.pulse_extraction import *
from src.tools import visualize
from src.tools.evaluate import *
from src.tools.opensignal import *
from src.tools.peak_detector import *
from src import preprocessing
from biosppy import signals
# vpath = r"./video/2020-05-29_static_imagesource.avi"
# landmark_data = r"./video/2020-05-29_static_imagesource.csv"
# outpath = "./result/rgb_2020-05-29_static_imagesource.csv"
# refpath = r"./video/2020-05-29_static_imagesource_opensignals.txt"

vpath = r"C:\Users\akito\Desktop\HassyLab\discussion\2020\2020-05-28\2020-05-29_static_imagesource.avi"
landmark_data = r"C:\Users\akito\Desktop\HassyLab\discussion\2020\2020-05-28\2020-05-29_static_imagesource.csv"
outpath = r"C:\Users\akito\Desktop\HassyLab\discussion\2020\2020-05-28\rgb_2020-05-29_static_imagesource.csv"
refpath = r"C:\Users\akito\Desktop\HassyLab\discussion\2020\2020-05-28\2020-05-29_opensignals_static_imagesource.txt"
delay = 5.0 # sec
length = 120 # sec

# 動画の読み込み
cap = cv2.VideoCapture(vpath)
# # Openfaceで取得したLandMark
df = pd.read_csv(landmark_data, header = 0).rename(columns=lambda x: x.replace(' ', ''))
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(df.shape)
ppg_signal = FaceAreaRoI(df,cap)
np.savetxt(outpath, ppg_signal, delimiter=",")

ref_ecg = np.loadtxt(refpath)[int(delay*1000):, -1]
ref_peaks = signals.ecg.ecg(ref_ecg, sampling_rate=1000, show=False)[2]

rgb_signal = np.loadtxt(outpath, delimiter=",")[int(delay*15):, :]
rppg_signal = POSMethod(rgb_signal, WinSec=1.6, fs=15, filter=False)

rppg_peaks = RppgPeakDetection(rppg_signal, fs=15, fr=250, filter=False, show=False, col=0.8)
rppg_peaks, rppg_rri = preprocessing.outlier_correction(rppg_peaks, threshold=0.10)

np.savetxt("ecg_result.csv", ref_ecg, delimiter=",")
np.savetxt("rppg_result.csv", rppg_signal, delimiter=",")


print("ref:{} est:{}".format(len(ref_peaks), len(rppg_peaks)))
fig,axes = plt.subplots(2, 1, sharex=True)
ts_ecg = np.arange(0,len(ref_ecg)/1000, 0.001)
ts_rppg = np.arange(0,len(rppg_signal)/15, 1/15)
axes[0].plot(ts_rppg, (rppg_signal-np.min(rppg_signal))/(np.max(rppg_signal)-np.min(rppg_signal)))
axes[0].plot(ts_ecg, (ref_ecg-np.min(ref_ecg))/(np.max(ref_ecg)-np.min(ref_ecg)))
axes[1].plot(ref_peaks[1:]/1000, preprocessing.RRInterval(ref_peaks))
axes[1].plot(rppg_peaks[1:], preprocessing.RRInterval(rppg_peaks)*1000)
plt.show()

# visualize.plot_PSD(ref_peaks[:len(rppg_peaks)]/1000, nfft=2**9, label="REF")
# visualize.plot_PSD(rppg_peaks, nfft=2**9, label="EST")
plt.plot(rppg_peaks - ref_peaks[:len(rppg_peaks)]/1000)
plt.show()
#filter_rppg = POSMethod(ppg_signal, WinSec=1.6, filter=False)

# visualize.plot_snr(ref_ppg[256:768])
# visualize.plot_snr(filter_rppg[256:768])
# visualize.plot_rri(ref_ppg, filter_rppg, filter_rppg)
# plt.show()

# # estimate
# t_T, HR_T = CalcTimeHR(rpeaks, rri, segment=17.06)
# t_F, HR_F = CalcFreqHR(filter_rppg)
# # reference
# ref_t_T, ref_HR_T = CalcTimeHR(ref_rpeaks, ref_rri, segment=17.06)
# ref_t_F, ref_HR_F = CalcFreqHR(ref_ppg)

# #plt.plot(ref_t_F, ref_HR_F, label="PPG FreqHR")
# plt.plot(ref_t_T, ref_HR_T, label="PPG TimeHR")
# plt.plot(t_F, HR_F, label="RPPG FreqHR")
# plt.plot(t_T, HR_T, label="RPPG TimeHR")
# plt.xlabel("Time[s]")
# plt.ylabel("HR [bpm]")

# plt.legend()

# print("MAE: {}".format(np.mean(abs(ref_HR_F-HR_F))))
# print("RMSE: {}".format(np.sqrt(np.mean((ref_HR_F-HR_F)**2))))
# print("Cov: {}".format(np.corrcoef(ref_HR_F,HR_F)[0, 1]))
# plt.show()


# nonfilter_rppg = POSMethod(ppg_signal, WinSec=12.8, filter=False)
# # nonfilter_rppg = GreenMethod(ppg_signal)
# filter_peaks = RppgPeakDetection(filter_rppg, fr=250)
# nonfilter_peaks = RppgPeakDetection(nonfilter_rppg, fr=250, filter=True)
# fil_peaks, fil_rri = preprocessing.outlier_correction(filter_peaks, threshold=0.13)
# visualize.plot_PSD(fil_peaks, fil_rri, label="Filter")
# visualize.plot_PSD(ref_peaks, label= "PPG")
# plt.legend()
# plt.show()
#visualize.plot_PSD(filter_peaks, label="Filter")
# visualize.plot_rri(ref_ppg, filter_rppg, nonfilter_rppg)

# rp,ri = preprocessing.outlier_correction(filter_peaks, threshold=0.15)

# HF,_ = CalcSNR(ref_ppg[300:])
# filter_snr = CalcSNR(filter_rppg[300:], HF)

# nonfilter_snr = CalcSNR(nonfilter_rppg[300:], HF)
# print("SNR: Filter={}  nonFilter={}".format(filter_snr, nonfilter_snr))

# visualize.plot_BlandAltman(filter_peaks[:int(ref_peaks.shape[0])],
#                            nonfilter_peaks[:int(ref_peaks.shape[0])],
#                            ref_peaks)
