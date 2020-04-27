# -*- coding: utf-8 -*-
import cv2


cap = cv2.VideoCapture(1)
for num in range(22):
    print(num, '.',cap.get(num))