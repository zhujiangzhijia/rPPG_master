"""
demo 
"""
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from src.pulse_extraction import *
filepath = r"./video/landmark_2020-04-30_motion_yaw.avi"
rgbpath = r"./result/rgb_2020-04-30_motion_yaw.csv"

sns.set()
# video設定
cap = cv2.VideoCapture(filepath)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('demo_2020-04-30_motion_yaw.mp4', fourcc, 30.0, (1200, 600))

# 歯入ってるっぽい
# いろいろ体裁を整える


# rPPG設定
rgb = np.loadtxt(rgbpath, delimiter=",")
ppg_green = GreenMethod(rgb)
ppg_pos = POSMethod(rgb)
ts = np.linspace(0, 30, num=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    print("Frame: {}".format(i))
    ret, frame = cap.read()
    frame = frame[:, 420:1501]
    frame = cv2.resize(frame, (600, 600)) 
    
    # 描画
    fig,axis = plt.subplots(2,1,sharex=True,figsize=(6,6))
    axis[0].plot(ts[:i+1], ppg_green[:i+1])
    axis[0].set_title("Green Method")
    axis[1].plot(ts[:i+1], ppg_pos[:i+1])
    axis[1].set_title("POS Method")
    axis[1].set_xlabel("Time[s]")
    if ts[i]<=5:
        plt.xlim(0,5)
    else:
        plt.xlim(ts[i]-5.0,ts[i])
    fig.canvas.draw()

    # VideoWriterへ書き込み
    image_array = np.array(fig.canvas.renderer.buffer_rgba())
    im = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    concat_frame = cv2.hconcat([frame,im])
    video.write(concat_frame)
    cv2.imshow("test",concat_frame)
    plt.close()
    cv2.waitKey(25)

cap.release()
video.release()
cv2.destroyAllWindows()