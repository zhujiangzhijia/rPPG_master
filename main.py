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
import time
vpath = r"./video/2020-05-11_motion_rotation.avi"
landmark_data = r"./video/2020-05-11_motion_rotation.csv"
outpath = "./result/rgb_2020-05-11_motion_rotation.csv"
refpath = "./video/2020-05-11_motion_rotation_ppg.csv"


# #動画の読み込み
# cap = cv2.VideoCapture(vpath)
# #Openfaceで取得したLandMark
# df = pd.read_csv(landmark_data, header = 0).rename(columns=lambda x: x.replace(' ', ''))
# print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(df.shape)
# ppg_signal = FaceAreaRoI(df,cap)
# np.savetxt(outpath, ppg_signal, delimiter=",")


ref_ppg = (np.loadtxt(refpath,delimiter=",")[:, 1] / 1023) * 5
ppg_signal = np.loadtxt(outpath,delimiter=",")
filter_pos = POSMethod(ppg_signal,WinSec=1.6,filter=False)
ts = np.loadtxt(refpath,delimiter=",")[:, 0]
rppg_peaks = RppgPeakDetection(filter_pos, fr=250)
ref_peaks = RppgPeakDetection(ref_ppg, fr=250)
ref_peaks = RppgPeakCorrection(ref_peaks)
plt.plot(ref_peaks[1:], preprocessing.RRInterval(rppg_peaks)[:ref_peaks.size-1], label="rPPG")
plt.plot(ref_peaks[1:], preprocessing.RRInterval(ref_peaks), label="PPG")
plt.legend()
# np.savetxt(r"rppg.csv", rppg_peaks, delimiter=",")
# np.savetxt(r"ppg.csv", ref_peaks, delimiter=",")
plt.show()




# # iPPG
# rppg_ts = np.loadtxt(r"C:\Users\akito\Desktop\HassyLab\programs\rPPG_master\video\2020-05_06_motion_yaw_timestamp.csv", delimiter=",")
# import datetime
# print(datetime.datetime.fromtimestamp(rppg_ts[0]))
# rppg_ts = rppg_ts - rppg_ts[0]
# rgb_signal = np.loadtxt(outpath, delimiter=",")
# ppg_signal = POSMethod(rgb_signal)
# rpeaks = RppgPeakDetection(ppg_signal,ts= rppg_ts ,fs=30, fr=100)
# rri = preprocessing.RRInterval(rpeaks)
# hr = 60 / rri


# # plt.plot(rpeaks[1:]-rpeaks[1], hr, label="rppg")
# # plt.plot(ecg_result['heart_rate_ts']-ecg_result['heart_rate_ts'][0], ecg_result['heart_rate'], label="ecg")
# # plt.legend()




# t_interpol, resamp_rppg = resampling(rppg_ts, POSMethod(ppg_signal), fs=30, fr=100)

# # リファレンス
# ecgdata = ImportECG(refpath)
# from biosppy.signals import ecg
# ecg_result = ecg.ecg(ecgdata, sampling_rate = 100., show=False)

# delay_time = rpeaks[5] - ecg_result['heart_rate_ts'][5]
# print(delay_time)

# fig,axes = plt.subplots(2,1,sharex=True)
# axes[0].plot(ecg_result["ts"],ecg_result["filtered"],label="ECG")
# axes[1].plot(t_interpol-delay_time, resamp_rppg,label="RPPG")
# plt.legend()
# plt.show()