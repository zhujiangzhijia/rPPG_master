"""
This  program is used to record the video
by using the web cam (logicool C920)
"""
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
fname = "2020-05_01_static"
csv

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'HFYU'))
cap.set(cv2.CAP_PROP_TEMPERATURE, 4000)
# # Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'HFYU')
out = cv2.VideoWriter("{}.avi".format(fname), fourcc, 30.0, (1920, 1080))


for num in range(47):
    print(num, '.', cap.get(num))

time_s = 30 # [s]
num_frame = time_s*30
print("FPSの設定値、video.get(cv2.CAP_PROP_FPS) : {0}".format(30.0))
print("解像度 : {} × {} ".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("取得中 {0} frames".format(num_frame)) 
start = time.time()
timestamp = np.array([[]])
for i in range(num_frame):
    ret, frame = cap.read()
    if ret == True:
        timestamp = np.append(timestamp, time.time())
        # write the flipped frame
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break  
np.savetxt("{}_timestamp.csv".format(fname),timestamp,delimiter=",")
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()