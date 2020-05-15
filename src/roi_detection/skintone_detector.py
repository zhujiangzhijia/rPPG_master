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
    HSV_MIN = np.array([0, 30, 60])
    HSV_MAX = np.array([20, 150, 255])
    # convert BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #mask hsv region
    mask_hsv = cv2.inRange(img_hsv, HSV_MIN, HSV_MAX)
    return mask_hsv


def SkinDetectYCbCr(img):
    """
    RGB to YCbCr
    """
    #YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCbCr_MIN = np.array([120, 143, 100])
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
 
    # read input image
    img = cv2.imread(r"C:\Users\akito\source\WebcamRecorder\output\UmcompressedVideo_2_aligned\frame_det_00_000371.bmp")
 
    #convert YCbCr
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    #img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    h, s, v = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]

    hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
    hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
    hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
    plt.plot(hist_h, color='r', label="h")
    plt.plot(hist_s, color='g', label="s")
    plt.plot(hist_v, color='b', label="v")
    plt.legend()
    #mask hsv region
    mask_hsv = cv2.inRange(img_hsv, HSV_MIN, HSV_MAX)
 
    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask = mask_hsv)
    cv2.imwrite("masked.png",mask_hsv)
    cv2.imwrite("masked_face.png",masked_img)
