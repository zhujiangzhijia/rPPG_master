from src.LandmarkExtract import *
import cv2
import pandas as pd


# vpath = r"C:\Users\akito\source\WebcamRecorder\output\UmcompressedVideo_2.avi"
# landmark_data = r"C:\Users\akito\source\WebcamRecorder\output\UmcompressedVideo_2.csv"

# #動画の読み込み
# cap = cv2.VideoCapture(vpath)

# #Openfaceで取得したLandMark
# df = pd.read_csv(landmark_data,header = 0).rename(columns=lambda x: x.replace(' ', ''))
# print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(df.shape)


# # RGB成分をROIから取り出す
# fpath = r"./result/rppg_result_umcomp_2.csv"
# ExportRGBComponents(df,cap,fpath)


from src.pulse_extraction.rPPG_CHROM import ChromMethod
from src.pulse_extraction.rPPG_GREEN import GreenMethod
from src.tools import visualize
import matplotlib.pyplot as plt
bgr_component = pd.read_csv("./result/rppg_result_umcomp_2.csv",
                            usecols=range(4), header=0,index_col=0).values
rgb_components = bgr_component[:, ::-1]
green_rppg = GreenMethod(rgb_components)
chrom_rppg = ChromMethod(rgb_components)

# visualize.plot_snr(green_rppg[100:], fs=30)
# visualize.plot_snr(chrom_rppg[100:], fs=30)

_,axes = plt.subplots(2,1,sharex=True)
axes[0].plot(green_rppg)
#axes[0].set_ylim(-1.5,1.5)
axes[1].plot(chrom_rppg)
plt.show()
