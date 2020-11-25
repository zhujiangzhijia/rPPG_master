"""
肌色検出
"""
# -*- coding: utf-8 -*-
import cv2
import numpy as np


def SkinDetectHSV(img):
    """
    RGB to HSV
    """
    HSV_MIN = np.array([0,29, 125])
    HSV_MAX = np.array([22,233, 255])
    # convert BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #mask hsv region
    mask_hsv = cv2.inRange(img_hsv, HSV_MIN, HSV_MAX)

    #ノイズ除去
    # # # 近傍の定義
    # element8 = np.ones((3,3),np.uint8)
    # mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, element8)
    # mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, element8)

    return mask_hsv


def SkinDetectYCbCr(img):
    """
    RGB to YCbCr
    """
    #YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCbCr_MIN = np.array([85, 110, 85])
    YCbCr_MAX = np.array([255, 180, 135])
    # convert BGR to yCbCr    
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #mask ycbcr region
    mask_ycbcr = cv2.inRange(img_ycbcr, YCbCr_MIN, YCbCr_MAX)

    return mask_ycbcr



if __name__ == "__main__":
    #define parameter
    # HSV_MIN = np.array([90, 133, 77])
    # HSV_MAX = np.array([255, 173, 127])
    HSV_MIN = np.array([150, 133, 77])
    HSV_MAX = np.array([255, 173, 127])
    path = r"D:\luminance_100lux\2020-09-14 17-23-19.125_Cam 1_1079_1078.bmp"
    # read input image
    img = cv2.imread(path)
    masked_img = SkinDetectHSV(img)
    # マスキング処理
    masked_img2 = cv2.bitwise_and(img, img, mask = masked_img)
    cv2.imshow("test", masked_img2)
    cv2.waitKey(0)
