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


vpath = r"video/2020-05_06_motion_rotation.avi"
landmark_data = r"video/2020-05_06_motion_rotation.csv"
outpath = "./result/rgb_2020-05-06_motion_rotation.csv"

# #動画の読み込み
# cap = cv2.VideoCapture(vpath)
# #Openfaceで取得したLandMark
# df = pd.read_csv(landmark_data, header = 0).rename(columns=lambda x: x.replace(' ', ''))
# print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(df.shape)
# ppg_signal= FaceAreaRoI(df,cap)
# np.savetxt(outpath ,ppg_signal,delimiter=",")




# iPPG
ppg_signal = np.loadtxt(outpath, delimiter=",")
rppg = GreenMethod(ppg_signal)
plt.plot(rppg)

# rppg_ts = np.loadtxt("./video/2020-05_06_motion_rotation_timestamp.csv",delimiter=",")
# import datetime
# print(datetime.datetime.fromtimestamp(rppg_ts[0]))
# rppg_ts = rppg_ts - rppg_ts[0]

# # リファレンス
# path = "./video/motionrotation_201808080163_2020-05-06_17-07-14.txt"
# ecgdata = ECG(path)
# ecg_ts = np.arange(0, rppg_ts[-1], 0.01)
# ecgdata = ecgdata[:len(ecg_ts)]


# fig,axes = plt.subplots(2, 1, sharex=True)
# axes[0].plot(ecg_ts, ecgdata)
# axes[1].plot(rppg_ts, rppg)


# fig, axes = plt.subplots(3,1,sharex=True, figsize=(16,9))
# axes[0].plot(GreenMethod(ppg_signal))
# axes[0].set_title("Green method")
# axes[1].plot(ChromMethod(ppg_signal))
# axes[1].set_title("Chrom method")
# axes[2].plot(POSMethod(ppg_signal))
# axes[2].set_title("POS method")
# axes[2].set_xlabel("Frame [-]")
# plt.legend()
plt.show()