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
vpath = r"./video/2020-05-29_static_imagesource.avi"
landmark_data = r"./video/2020-05-29_static_imagesource.csv"
outpath = "./result/rgb_2020-05-11_motion_talking.csv"
refpath = r"./video/2020-05-29_static_imagesource_opensignals.txt"
delay = 5.0 # sec
length = 120 # sec

# # -------------動画の読み込み--------------
# cap = cv2.VideoCapture(vpath)
# # # Openfaceで取得したLandMark
# df = pd.read_csv(landmark_data, header = 0).rename(columns=lambda x: x.replace(' ', ''))
# print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(df.shape)
# ppg_signal = FaceAreaRoI(df,cap)
# np.savetxt(outpath, ppg_signal, delimiter=",")
# ref_ecg = np.loadtxt(refpath)[int(delay*1000):, -1]
# ref_peaks = signals.ecg.ecg(ref_ecg, sampling_rate=1000, show=False)[2]



rgb_signal = np.loadtxt(outpath, delimiter=",")
rppg_pos = POSMethod(rgb_signal, WinSec=1.6, fs=30, filter=False)
rppg_green = GreenMethod(rgb_signal, fs=30)
result = np.concatenate((rppg_pos.reshape(-1,1),rppg_green.reshape(-1,1)), axis=1)
np.savetxt(r"C:\Users\akito\Desktop\rppg_motion_talking.csv",result,delimiter=",")
_, axes= plt.subplots(2,1,sharex=True)
axes[0].plot(rppg_pos)
axes[1].plot(rppg_green)
plt.show()

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


