"""
マウス操作
"""
# -*- coding: utf-8 -*-
import cv2
import numpy as np

winname = 'Image'
filename = r"C:\Users\akito\Desktop\Cam 1_100fps\2020-07-09 16-51-34.564_Cam 1_1_0.bmp"

image = cv2.imread(filename)

if image is None:
   print('can''t open ' + filename)
   raise RuntimeError

# x,y,w,h
rois = cv2.selectROIs(winname, image, False)

for r in rois:
   cv2.rectangle(image,(r[0],r[1]),(r[2]+r[0],r[3]+r[1]), (255, 0, 0),thickness=1)
# Crop image
# imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv2.imshow(winname, image)
cv2.waitKey(0)


# (r[1],r[1]+r[3]), (r[0],r[0]+r[2])