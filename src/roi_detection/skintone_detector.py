"""
肌色検出
"""
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
#3次元プロットするためのモジュール
from mpl_toolkits.mplot3d import Axes3D

def SkinDetect(img):
    mask_hsv = SkinDetectHSV(img)
    mask_ycbcr = SkinDetectYCbCr(img)
    mask_all = cv2.bitwise_and(mask_ycbcr, mask_ycbcr, mask_hsv)
    # skinmask = cv2.bitwise_and(img, img, mask=mask_all)
    mask_all = ReduceNoise(mask_all)
    return mask_all

def SkinDetectTrack(img,path):
    """
    手動で閾値設定
    """
    YCbCr_MIN,YCbCr_MAX = np.load(path)

    # convert BGR to yCbCr    
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #mask ycbcr region
    mask_ycbcr = cv2.inRange(img_ycbcr, YCbCr_MIN, YCbCr_MAX)
    return mask_ycbcr


def SkinDetectHSV(img,auto=True):
    """
    RGB to HSV
    """
    
    HSV_MIN = np.array([0, 40, 0])
    HSV_MAX = np.array([25, 255, 255])
    # convert BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if auto:
        #大津の二値化
        auto_threshold,_ = cv2.threshold(img_hsv[:,:,0],0,255,cv2.THRESH_OTSU)
        HSV_MAX[0] =  auto_threshold
        print("threshold: {}".format(auto_threshold))

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

def SkinDetecOneClassSVM(img):
    """
    One Class SVMを使った皮膚領域抽出手法
    """
    # convert BGR to yCbCr    
    arr_origin_bgr = img.reshape(-1,3)
    
    # 顔領域のみを抽出
    flag = np.all(arr_origin_bgr != 0, axis=1)
    arr_bgr = arr_origin_bgr[flag,:]
    extract_ratio  = 100* (arr_bgr.shape[0]/arr_bgr.shape[0] ) # 画像内における抽出領域の割合
    print(extract_ratio)

    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    arr_ycbcr = img_ycbcr.reshape(-1,3)[flag,:]
    
    # 標準化
    scaler = preprocessing.StandardScaler()
    standard_arr_ycbcr = scaler.fit_transform(arr_ycbcr)
    # SVMによる外れ値検出
    clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto') # nuが外れ値の割合
    clf.fit(standard_arr_ycbcr)
    pred = clf.predict(standard_arr_ycbcr)

    # 画像に戻す
    arr_index = np.where(flag == True)[0] # RoI 顔と検出された領域
    arr2 = (pred < 0) # 正常: 1、異常: -1
    out_index = arr_index[arr2] # 肌色検出で肌でないとされた部分
    flag[out_index] = 0


    test = arr_origin_bgr == np.array([[0,0,0]])
    img_mask = (arr_origin_bgr * flag.reshape(-1,1)).reshape(img.shape)

    # マスク画像作成
    # 画像の黒い部分を白に置き換える
    black = [0, 0, 0]
    white = [255, 255, 255]
    img_mask[np.where((img_mask == black).all(axis=2))] = white


    # test_img_mask = np.where(((img_mask[:,:,0]== 0) & (img_mask[:,:,1== 0) & (img_mask[:,:,2]== 0)),)
    cv2.imshow("RoI Detect", img_mask)
    

    cv2.waitKey(0)


    arr_bgr = img.reshape(-1, 3)


    # arr_ycbcrの配列内のFlagを変換する



    # #グラフの枠を作っていく
    # fig = plt.figure()
    # ax = Axes3D(fig)

    # #軸にラベルを付けたいときは書く
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    # #.plotで描画
    # #linestyle='None'にしないと初期値では線が引かれるが、3次元の散布図だと大抵ジャマになる
    # #markerは無難に丸
    # ax.plot(arr_ycbcr[:,0],arr_ycbcr[:,1],arr_ycbcr[:,2],marker="o",linestyle='None')

    # #最後に.show()を書いてグラフ表示
    # plt.show()

    return 0 
    

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
    path = r"D:\rPPGDataset\Figure\Images\shizuya\luminance\roimask\2021-01-05 18-06-02.745698 Front 100lux Cam.bmp"

    img = cv2.imread(path)
    SkinDetecOneClassSVM(img)

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
