from src.LandmarkExtract import *
import cv2
import pandas as pd


vpath = r"C:\Users\akito\source\WebcamRecorder\UmcompressedVideo_origin.avi"
landmark_data = r"C:\Users\akito\source\WebcamRecorder\output\UmcompressedVideo_origin.csv"

#動画の読み込み
cap = cv2.VideoCapture(vpath)

#Openfaceで取得したLandMark
df = pd.read_csv(landmark_data,header = 0,usecols=range(299,434)).rename(columns=lambda x: x.replace(' ', ''))
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(df.shape)

# RGB成分をROIから取り出す
fpath = r"C:\Users\akito\Desktop\result_2020-04-26_tohma.csv"
ExportRGBComponents(df,cap,fpath)


