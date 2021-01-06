"""
demo 
"""
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from src.pulse_extraction import *
filepath = r"D:\20201125\exp2\output\exp2.avi"
rgbpath = r"D:\20201125\exp2\output\rgb_signal_exp2.csv"
fps = 40

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
# ppg_pos = GreenMethod(rgb)
ppg_pos = POSMethod(rgb,fs=fps,filter=True)
ts = np.arange(0,(len(ppg_pos)/fps),1/fps)

start = 10*fps
end = 40*fps

for i in range(start,end):
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

    plt.plot(ts[int(start):i+1]-ts[int(start)], ppg_pos[int(start):i+1])
    plt.title("Estimation")
    plt.xlabel("Time[s]")
    if np.max(ts[int(start):i+1]-ts[int(start)])<=5:
        plt.xlim(0,5)
    else:
        plt.xlim(np.max(ts[int(start):i+1]-ts[int(start)])-5.0,np.max(ts[int(start):i+1]-ts[int(start)]))
    fig.canvas.draw()

    # VideoWriterへ書き込み
    # image_array = np.array(fig.canvas.renderer.buffer_rgba())
    image_array = np.array(fig.canvas.renderer._renderer) # matplotlibが3.1より前の場合


    im = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    concat_frame = cv2.hconcat([frame,im])
    video.write(concat_frame)
    plt.close()
    cv2.imshow("test",concat_frame)
    cv2.waitKey(25)

cap.release()
video.release()
cv2.destroyAllWindows()