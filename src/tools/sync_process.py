"""
同期用
"""
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

vpath = r"C:\Users\akito\Desktop\HassyLab\discussion\2020\2020-05-28\2020-05-29_static_imagesource.avi"
cap = cv2.VideoCapture(vpath)
data = np.array([[]])
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print(fourcc)
#out = cv2.VideoWriter("2020-05-28_static_imagesource_trim.avi", fourcc, 20, (640, 480))


# loop from first to last frame
for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ret, frame = cap.read()
    B_value, G_value, R_value = frame.T
    rgb_component = np.array([[np.mean(R_value), np.mean(G_value), np.mean(B_value)]])
    # if i >= 1000:
    #     out.write(frame)
    if i == 0:
        data = rgb_component
    else:
        data = np.concatenate((data,rgb_component),axis=0)

cap.release()
# out.release()
cv2.destroyAllWindows()

plt.plot(data[1:, 1]-data[:-1, 1])
plt.show()
