"""
実行ファイル
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from src.roi_detection.landmark_extractor import *
import cv2

from src.pulse_extraction.rPPG_CHROM import ChromMethod
from src.pulse_extraction.rPPG_GREEN import GreenMethod
from src.pulse_extraction.rPPG_POS import POSMethod
from src.tools import visualize
import matplotlib.pyplot as plt


# vpath = r"C:\Users\akito\source\WebcamRecorder\output\UmcompressedVideo_3.avi"
# landmark_data = r"C:\Users\akito\source\WebcamRecorder\output\UmcompressedVideo_3.csv"

# #動画の読み込み
# cap = cv2.VideoCapture(vpath)

# #Openfaceで取得したLandMark
# df = pd.read_csv(landmark_data,header = 0).rename(columns=lambda x: x.replace(' ', ''))
# print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(df.shape)


# # RGB成分をROIから取り出す
# fpath = r"./result/rppg_result_umcomp_3.csv"
# ExportRGBComponents(df,cap,fpath)



rgb_component = np.read_csv("./result/rgb_ucomp2_faceroi.csv", delimiter=",")
# rppg1 = POSMethod(rgb_components)
# rppg2 = ChromMethod(rgb_components)
rppg = GreenMethod(rgb_components)
#plt.plot(rppg1,label="POS")
plt.plot(rppg,label="GREEN")

#plt.plot(rppg3,label="GREEN")
plt.legend()
plt.show()