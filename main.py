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


vpath = r"C:\Users\akito\Desktop\HassyLab\programs\rPPG_master\video\2020-04-30_motion_yaw.avi"
landmark_data = r"C:\Users\akito\Desktop\HassyLab\programs\rPPG_master\video\2020-04-30_motion_yaw.csv"
outpath = "./result/rgb_2020-04-30_motion_taking.csv"

# #動画の読み込み
cap = cv2.VideoCapture(vpath)

#Openfaceで取得したLandMark
df = pd.read_csv(landmark_data,header = 0).rename(columns=lambda x: x.replace(' ', ''))
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(df.shape)

data = FaceAreaRoI(df,cap)
np.savetxt(outpath ,data,delimiter=",")

rgb_components = np.loadtxt(outpath, delimiter=",")
fig, axes = plt.subplots(3,1,sharex=True)

axes[0].plot(GreenMethod(rgb_components))
axes[0].set_title("Green method")
axes[1].plot(ChromMethod(rgb_components))
axes[1].set_title("Chrom method")
axes[2].plot(POSMethod(rgb_components))
axes[2].set_title("POS method")
plt.legend()
plt.show()