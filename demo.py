"""
demo 
"""
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from src.pulse_extraction import *
filepath = r"D:\RppgDatasets\04.Video\nonfixed\double\Cam 1\output\Landmark.avi"
rgbpath = r"D:\RppgDatasets\03.BiometricData\LightSource\no_restraint_celling_and_front\rgb_no_restraint_celling_and_front.csv"

sns.set()
# video設定
cap = cv2.VideoCapture(filepath)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  #fourccを定義
# video = cv2.VideoWriter('demo_stationary.mp4', fourcc, 100.0, (720, 540))
video = cv2.VideoWriter('demo_stationary.mp4', fourcc, 40.0, (1200, 600))

# rPPG設定
rgb = np.loadtxt(rgbpath, delimiter=",")
# ppg_green = GreenMethod(rgb)
ppg_pos = GreenMethod(rgb)
ts = np.arange(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)*0.01),0.01)
for i in range(300,1800):
    print("Frame: {}".format(i))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    frame = frame[:, :]
    frame = cv2.resize(frame, (600, 600)) 

    # 描画
    #fig,axis = plt.subplots(1,1,sharex=True,figsize=(6,6))
    fig = plt.figure(figsize=(6,6))

    # axis[0].plot(ts[:i+1], ppg_green[:i+1])
    # axis[0].set_title("Green Method")
    plt.plot(ts[300:i+1]-ts[300], ppg_pos[300:i+1])
    plt.title("Estimation")
    plt.xlabel("Time[s]")
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
    plt.close()
    # cv2.imshow("test",concat_frame)
    # cv2.waitKey(25)

cap.release()
video.release()
cv2.destroyAllWindows()