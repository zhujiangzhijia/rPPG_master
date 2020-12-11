"""
肌色検出
"""
# -*- coding: utf-8 -*-
import cv2
import numpy as np


def SkinDetect(img):
    mask_hsv = SkinDetectHSV(img)
    mask_ycbcr = SkinDetectYCbCr(img)
    mask_all = cv2.bitwise_and(mask_ycbcr, mask_ycbcr, mask_hsv)
    # skinmask = cv2.bitwise_and(img, img, mask=mask_all)
    mask_all = ReduceNoise(mask_all)
    return mask_all

def SkinDetectHSV(img):
    """
    RGB to HSV
    """
    HSV_MIN = np.array([0, 40, 0])
    HSV_MAX = np.array([25, 255, 255])
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
    YCbCr_MIN = np.array([0, 138, 67])
    YCbCr_MAX = np.array([255, 173, 133])
    # convert BGR to yCbCr    
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #mask ycbcr region
    mask_ycbcr = cv2.inRange(img_ycbcr, YCbCr_MIN, YCbCr_MAX)

    return mask_ycbcr

def ReduceNoise(img):
    # ノイズ除去(膨張・収縮)
	# using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    skinMask = cv2.erode(img, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
	# ガウシアンフィルタにより，ノイズを抑える
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    return skinMask

def SkinDetectTEST(img):
    cv2.imshow("img",img)
    mask_hsv = SkinDetectHSV(img)
    mask_ycbcr = SkinDetectYCbCr(img)
    cv2.imshow("HSV", mask_hsv)
    cv2.imshow("YCbCr",mask_ycbcr)
    skinmask_ycbcr = cv2.bitwise_and(img, img, mask=mask_ycbcr)
    cv2.imshow("mask_Ycbcr",skinmask_ycbcr)
    skinmask_hsv = cv2.bitwise_and(img, img, mask=mask_hsv)
    cv2.imshow("mask_hsv", skinmask_hsv)
    mask_all = cv2.bitwise_and(mask_ycbcr, mask_ycbcr, mask_hsv)
    cv2.imshow("mask_all", skinmask)
    skinmask = ReduceNoise(skinmask)
    cv2.imshow("mask_all_Renoise", skinmask)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    path = r"D:\RppgDatasets\no_restraint_front_2\2020-09-14 16-15-38.057_Cam 1_4811_4811.bmp"
    img = cv2.imread(path)
    SkinDetect(img)

    # #define parameter
    # # HSV_MIN = np.array([90, 133, 77])
    # # HSV_MAX = np.array([255, 173, 127])
    # HSV_MIN = np.array([153, 63, 90])
    # HSV_MAX = np.array([180, 255, 255])
    # path = r"C:\Users\akito\Desktop\2020-09-14 17-46-37.680_Cam 1_724_723.bmp"
    # # read input image
    # img = cv2.imread(path)
    # masked_img = SkinDetectHSV(img)
    # # マスキング処理
    # masked_img2 = cv2.bitwise_and(img, img, mask = masked_img)
    # cv2.imshow("test", masked_img2)
    # cv2.waitKey(0)
