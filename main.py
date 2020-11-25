"""
実行ファイル
"""
# -*- coding: utf-8 -*-
import os
import re
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
import config as cf

# files = []

# i = 0
# for filename in os.listdir(cf.DIR_PATH):
#     if os.path.isfile(os.path.join(cf.DIR_PATH, filename)): #ファイルのみ取得
#         files.append(filename)

# # ファイル名からフレーム時刻を取得
# timestamps = []
# for file in files:
#     file_param = re.split('[_ ]', file)
#     timestamp = re.split('[-]', file_param[1])
#     # Str to int
#     timestamp = [float(s) for s in timestamp]
#     timestamps.append(timestamp[0]*60**2 + timestamp[1]*60 + timestamp[2])

# data_time = np.array(timestamps)
# data_time = data_time - data_time[0]
# data_timediff = 1/np.diff(data_time)
# print(np.mean(data_timediff))


# -------------動画の読み込み--------------
# df = pd.read_csv(cf.LANDMARK_PATH, header = 0).rename(columns=lambda x: x.replace(' ', ''))
# rgb_signal = FaceAreaRoI(df,cf.DIR_PATH)
cap = cv2.VideoCapture(cf.DIR_PATH)
# rgb_signal = FaceAreaRoIVideo(df, cap)
rgb_signal = MouseRoIVideo(cap)
np.savetxt(cf.OUTPUT_PATH, rgb_signal, delimiter=",")
exit()
# RPPG
rgb_signal = np.loadtxt(cf.OUTPUT_PATH, delimiter=",")
rppg_pos = POSMethod(rgb_signal, fs=52, filter=False)
rppg_green = GreenMethod(rgb_signal,fs=52)
rppg_chrom = ChromMethod(rgb_signal, fs=52)
_,axes = plt.subplots(3,1,sharex=True)
axes[0].plot(rppg_pos)
axes[1].plot(rppg_green)
axes[2].plot(rppg_chrom)

plt.show()

est_rpeaks = RppgPeakDetection(-rppg_pos,fs=30,show=True,col=0.15,filter=True)
plt.show()
exit()


# Cut-Time
rppg_pos = rppg_pos[(data_time>cf.DELAY_T) & (data_time<cf.DURATION_T+cf.DELAY_T)]
data_time = data_time[(data_time>cf.DELAY_T) & (data_time<cf.DURATION_T+cf.DELAY_T)]-cf.DELAY_T
est_rpeaks = RppgPeakDetection(rppg_pos,data_time, show=True, filter=True,col=0.4)

# Reference
ref_ecg = np.loadtxt(cf.REF_PATH)[int(cf.DELAY_T*1000):int((cf.DURATION_T+cf.DELAY_T)*1000), -2]
ref_peaks = signals.ecg.ecg(ref_ecg, sampling_rate=1000, show=False)[2]
ts = np.arange(0, len(ref_ecg)*0.001, 0.001)
plt.plot(ref_peaks[1:], ref_peaks[1:]-ref_peaks[:-1],label="REF")
plt.plot(est_rpeaks[1:], est_rpeaks[1:]-est_rpeaks[:-1],label="EST")
plt.legend()
plt.show()

est_rpeaks = est_rpeaks[:len(ref_peaks)]
visualize.plot_BlandAltman(est_rpeaks*0.001,ref_peaks*0.001)


result = np.concatenate((ref_peaks.reshape(-1,1),
                         est_rpeaks.reshape(-1,1)), axis=1)

# np.savetxt(r"D:\RppgDatasets\05.Analysis\LightSource\no_restraint_celling_peaks.csv",result,delimiter=",")


# #plt.legend()
import pyhrv
pyhrv.frequency_domain.welch_psd(nni = est_rpeaks[1:] - est_rpeaks[:-1],show=True,detrend=False,nfft=2**9)
pyhrv.frequency_domain.welch_psd(nni = ref_peaks[1:] - ref_peaks[:-1], show=True,detrend=False,nfft=2**9)
visualize.plot_PSD(est_rpeaks*0.001, label="EST")
visualize.plot_PSD(ref_peaks*0.001, label="REF")
plt.show()

#_,axes = plt.subplots(2,1,sharex=True)
#axes[0].plot(data_time[data_time>cf.DELAY_T]-cf.DELAY_T,rppg_pos[data_time>cf.DELAY_T])
#axes[1].plot(ts, ref_ecg)
plt.show()

# cap = cv2.VideoCapture(cf.vpath)
# # # Openfaceで取得したLandMark
# df = pd.read_csv(cf.landmark_data, header = 0).rename(columns=lambda x: x.replace(' ', ''))

# print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(df.shape)
# ppg_signal = FaceAreaRoI(df,cap)
# np.savetxt(outpath, ppg_signal, delimiter=",")
# ref_ecg = np.loadtxt(refpath)[int(delay*1000):, -1]
# ref_peaks = signals.ecg.ecg(ref_ecg, sampling_rate=1000, show=False)[2]

# rgb_signal = np.loadtxt(outpath, delimiter=",")
# rppg_pos = POSMethod(rgb_signal, WinSec=1.6, fs=30, filter=False)
# rppg_green = GreenMethod(rgb_signal, fs=30)
# result = np.concatenate((rppg_pos.reshape(-1,1),rppg_green.reshape(-1,1)), axis=1)
# np.savetxt(r"C:\Users\akito\Desktop\rppg_motion_talking.csv",result,delimiter=",")
# _, axes= plt.subplots(2,1,sharex=True)
# axes[0].plot(rppg_pos)
# axes[1].plot(rppg_green)
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


