"""
RoI領域の抽出
"""
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import cv2
from . import skintone_detector as sd


def ShibataRoI(df,cap):
    """
    RoI領域を複数選択後，平均化されたRGBを返す
    """
    # 顔スキャンし初期化
    wide_face, wide_nose, hight_eye, hight_cheek, hight_Eyebrows, hignht_nose = ScanFaceSize(df)
    i = 0
    pix_x_frames = df.loc[:, df.columns.str.contains('x_')].values.astype(np.int)
    pix_y_frames = df.loc[:, df.columns.str.contains('y_')].values.astype(np.int)
    print(pix_x_frames.shape)
    print(pix_y_frames.shape)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter を作成する。
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    #writer = cv2.VideoWriter("20200426_test.avi", fourcc, fps, (width, height))


    allRGBArrays = None
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        print("Frame:{}".format(i))
        roi_list = []
        rgb_item = np.array([[]])
        ret, frame = cap.read()
        pix_x = pix_x_frames[i,:]
        pix_y = pix_y_frames[i,:]
        

        # xy座標の順番注意
        #　何しているのか不明．補正っぽい？
        y_list = [pix_y[27], pix_y[28], pix_y[29],pix_y[30],pix_y[31]]
        y_list = sorted(y_list)
        y_list = np.clip(y_list, 0, None)
        new_y_list=[]
        for idx in range(len(y_list)-1):
            if y_list[idx]>=y_list[idx+1]:
                y_list[idx+1]=y_list[idx]+1
            new_y_list.append(y_list[idx])

        new_y_list.append(y_list[-1])
        pix_y[27], pix_y[28], pix_y[29],pix_y[30],pix_y[31] = new_y_list

        # RoI領域の定義
        roi_list.append(frame[pix_y[29]:pix_y[29] + hight_cheek, pix_x[31] - wide_face:pix_x[31]])#1.右頬
        roi_list.append(frame[pix_y[29]:pix_y[29] + hight_cheek,  pix_x[35]:pix_x[35] + wide_face ])#2.左頬
        roi_list.append(frame[pix_y[27]:pix_y[30],   pix_x[29] - wide_nose:pix_x[29] + wide_nose])#3.鼻全体
        roi_list.append(frame[pix_y[27]:pix_y[28],  pix_x[29] - wide_nose:pix_x[29] + wide_nose])#4.鼻上
        roi_list.append(frame[pix_y[28]:pix_y[29],  pix_x[29] - wide_nose:pix_x[29] + wide_nose])#5.鼻真ん中
        roi_list.append(frame[pix_y[29]:pix_y[30],   pix_x[29] - wide_nose:pix_x[29] + wide_nose])#6.鼻下
        roi_list.append(frame[np.clip(pix_y[29] - hight_eye, 0, None):pix_y[29] + hight_cheek, pix_x[31] - wide_face:pix_x[31]])#7.右頬ワイド
        roi_list.append(frame[np.clip(pix_y[29] - hight_eye, 0, None):pix_y[29] + hight_cheek, pix_x[35]:pix_x[35] + wide_face ])#8.左頬ワイド
        roi_list.append(frame[pix_y[29]:pix_y[31],  pix_x[29] - wide_face:pix_x[29] + wide_face])#9.全体
        roi_list.append(frame[np.clip(pix_y[29] - hight_eye, 0, None):pix_y[31], pix_x[29] - wide_face:pix_x[29] + wide_face])#10.全体ワイド
        roi_list.append(frame[pix_y[29]:pix_y[30],  pix_x[29]- wide_face:pix_x[29] + wide_face])#11.全体スモール
        
        # RoI領域の描画
        plot_roi(frame,pix_y[29],pix_y[29] + hight_cheek, pix_x[31] - wide_face,pix_x[31])#1.右頬
        plot_roi(frame,pix_y[29],pix_y[29] + hight_cheek,  pix_x[35],pix_x[35] + wide_face )#2.左頬
        plot_roi(frame,pix_y[27],pix_y[30],   pix_x[29] - wide_nose,pix_x[29] + wide_nose)#3.鼻全体
        plot_roi(frame,pix_y[27],pix_y[28],  pix_x[29] - wide_nose,pix_x[29] + wide_nose)#4.鼻上
        plot_roi(frame,pix_y[28],pix_y[29],  pix_x[29] - wide_nose,pix_x[29] + wide_nose)#5.鼻真ん中
        plot_roi(frame,pix_y[29],pix_y[30],   pix_x[29] - wide_nose,pix_x[29] + wide_nose)#6.鼻下
        plot_roi(frame,np.clip(pix_y[29] - hight_eye, 0, None),pix_y[29] + hight_cheek, pix_x[31] - wide_face,pix_x[31])#7.右頬ワイド
        plot_roi(frame,np.clip(pix_y[29] - hight_eye, 0, None),pix_y[29] + hight_cheek, pix_x[35],pix_x[35] + wide_face)#8.左頬ワイド
        plot_roi(frame,pix_y[29],pix_y[31],  pix_x[29] - wide_face,pix_x[29] + wide_face)#9.全体
        plot_roi(frame,np.clip(pix_y[29] - hight_eye, 0, None),pix_y[31], pix_x[29] - wide_face,pix_x[29] + wide_face)#10.全体ワイド
        plot_roi(frame,pix_y[29],pix_y[30],  pix_x[29]- wide_face,pix_x[29] + wide_face)#11.全体スモール

        for roi in roi_list:
            rgb_component = AveragedRGB(roi)
            rgb_item = np.concatenate([rgb_item, rgb_component], axis=1)
        cv2.imshow('frame',frame)

        # マージ処理
        if allRGBArrays is None:
            allRGBArrays = rgb_item
        else:
            allRGBArrays = np.concatenate([allRGBArrays,rgb_item],axis=0)
        
        #writer.write(frame)  # フレームを書き込む。
        
        cv2.waitKey(25)

    #writer.release()
    cap.release()
    cv2.destroyAllWindows()
    return allRGBArrays
def plot_roi(frame,hight_top,hight_bottom,width_left,width_right):
    cv2.rectangle(frame, (width_right, hight_bottom), (width_left, hight_top), color=(0,0,255),thickness= 4)



def MultipleRoI(df, dirpath, skin_detection=True, pixsize=20):
    """
    openfaceのlandmarkを使って，顔領域を選択し平均化されたRGBを返す
    """
    # Import landmark 
    pix_x_frames = df.loc[:, df.columns.str.contains('x_')].values.astype(np.int)
    pix_y_frames = df.loc[:, df.columns.str.contains('y_')].values.astype(np.int)
    # ファイル一覧を取得
    files = []
    i = 0
    for filename in os.listdir(dirpath):
        if os.path.isfile(os.path.join(dirpath, filename)): #ファイルのみ取得
            files.append(filename)

    # loop from first to last frame
    for fname in files:
        frame = cv2.imread(os.path.join(dirpath, fname))
        pix_x = pix_x_frames[i,:].reshape(-1, 1)
        pix_y = pix_y_frames[i,:].reshape(-1, 1)

        # FaceMask by features point
        mask = RoIDetection(frame,pix_x,pix_y)
        face_img = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("frame1", face_img)

        # skin area detection HSV & YCbCr
        if skin_detection:
            skin_mask = sd.SkinDetect(face_img)
            mask = cv2.bitwise_and(mask, skin_mask, skin_mask)
            cv2.imshow("skin_mask", skin_mask)

        # merge the mask image
        # average bgr components
        mask_img = cv2.bitwise_and(frame, frame, mask=mask)
        ave_rgb = np.array(cv2.mean(frame, mask=mask)[::-1][1:]).reshape(1,-1)

        if i == 0:
            rgb_components = ave_rgb
        else:   
            rgb_components = np.concatenate([rgb_components, ave_rgb], axis=0)

        cv2.imshow("frame2", mask_img)
        i = i + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return rgb_components



def ScanFaceSize(df,initframe=200):
    """
    顔のサイズを初期化する
    """
    a = 0.7
    wide_face      = int((df.x_29[initframe]-df.x_2[initframe])*a)
    wide_nose      = int((df.x_29[initframe]-df.x_39[initframe])*a)
    hight_eye      = int((df.y_29[initframe]-df.y_40[initframe])*a)
    hight_cheek    = int((df.y_33[initframe]-df.y_29[initframe])*a)
    hight_Eyebrows = int((df.y_29[initframe]-df.y_27[initframe])*a)
    hignht_nose    = int((df.y_30[initframe]-df.y_29[initframe])*a)

    return wide_face,wide_nose,hight_eye,hight_cheek,hight_Eyebrows,hignht_nose


def FaceAreaRoI(df,filepath,skin_detection=True):
    """
    openfaceのlandmarkを使って，顔領域を選択し平均化されたRGBを返す
    """
    # Import landmark 
    pix_x_frames = df.loc[:, df.columns.str.contains('x_')].values.astype(np.int)
    pix_y_frames = df.loc[:, df.columns.str.contains('y_')].values.astype(np.int)

    if os.path.isfile(filepath):
        cap = cv2.VideoCapture(filepath)
        format_video = True
        total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        files = []
        for filename in os.listdir(filepath):
            if os.path.isfile(os.path.join(filepath, filename)): #ファイルのみ取得
                files.append(filename)
        total_frame_num = len(files)
        format_video = False
    
    # loop from first to last frame
    for i in range(total_frame_num):
        print("Frame: {}/{}".format(i,total_frame_num))
        pix_x = pix_x_frames[i,:].reshape(-1, 1)
        pix_y = pix_y_frames[i,:].reshape(-1, 1)

        if format_video:
            ret, frame = cap.read()
        else:
            frame = cv2.imread(os.path.join(filepath, files[i]))
            
        # FaceMask by features point
        mask = RoIDetection(frame,pix_x,pix_y)
        face_img = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("RoI Detect", face_img)

        # skin area detection HSV & YCbCr
        if skin_detection:
            skin_mask = sd.SkinDetect(face_img)
            mask = cv2.bitwise_and(mask, skin_mask, skin_mask)
            cv2.imshow("Skin Mask", skin_mask)

        # merge the mask image
        # average bgr components
        mask_img = cv2.bitwise_and(frame, frame, mask=mask)
        ave_rgb = np.array(cv2.mean(frame, mask=mask)[::-1][1:]).reshape(1,-1)

        if i == 0:
            rgb_components = ave_rgb
        else:   
            rgb_components = np.concatenate([rgb_components, ave_rgb], axis=0)

        cv2.imshow("Mask Img", mask_img)
        i = i + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return rgb_components



def MouseRoI(dirpath):
    """
    openfaceのlandmarkを使って，顔領域を選択し平均化されたRGBを返す
    """
    # ファイル一覧を取得
    files = []
    i = 0
    for filename in os.listdir(dirpath):
        if os.path.isfile(os.path.join(dirpath, filename)): #ファイルのみ取得
            files.append(filename)

    # マウスイベント
    winname = 'Image'
    image = cv2.imread(os.path.join(dirpath, files[816]))
    rois = cv2.selectROIs(winname, image, False) # x,y,w,h
    cv2.destroyAllWindows()
    for r in rois:
        print("x:{}, y:{}, w:{}, h:{}".format(r[0],r[1],r[2],r[3]))
 
    
    # loop from first to last frame
    for fname in files[816:]:
        frame = cv2.imread(os.path.join(dirpath, fname))
        j = 0
        for r in rois:
            img_roi = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            if j == 0:
                ave_rgb_roi = AveragedRGB(img_roi)
            else:
                ave_rgb_roi = np.concatenate([ave_rgb_roi,AveragedRGB(img_roi)], axis=1)
            j = j+1
    
        if i == 0:
            rgb_components = ave_rgb_roi
        else:   
            rgb_components = np.concatenate([rgb_components, ave_rgb_roi], axis=0)

        # plot
        for r in rois:
            cv2.rectangle(frame,(r[0],r[1]),(r[2]+r[0],r[3]+r[1]), (255, 0, 0),thickness=3)
        cv2.imshow("frame", frame)
        i = i + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return rgb_components


def MouseRoIVideo(cap):
    """
    openfaceのlandmarkを使って，顔領域を選択し平均化されたRGBを返す
    """
    # マウスイベント
    winname = 'Image'
    ret, image = cap.read()
    rois = cv2.selectROIs(winname, image, False) # x,y,w,h
    cv2.destroyAllWindows()
    for r in rois:
        print("x:{}, y:{}, w:{}, h:{}".format(r[0],r[1],r[2],r[3]))
 
    # loop from first to last frame
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1):
        j = 0
        #for fname in files:
        print("Frame: {}/{}".format(i,cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        ret, frame = cap.read()
        for r in rois:
            img_roi = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            if j == 0:
                ave_rgb_roi = AveragedRGB(img_roi)
            else:
                ave_rgb_roi = np.concatenate([ave_rgb_roi,AveragedRGB(img_roi)], axis=1)
            j = j+1
    
        if i == 0:
            rgb_components = ave_rgb_roi
        else:   
            rgb_components = np.concatenate([rgb_components, ave_rgb_roi], axis=0)

        # cv2.imshow("frame", mask_img)
        i = i + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(i)

    cv2.destroyAllWindows()
    return rgb_components

def RoIDetection(frame,pix_x,pix_y):
    height, width = frame.shape[:2]
    # roi segmentation
    landmarks = np.concatenate([pix_x, pix_y],axis=1)
    white_img = np.zeros((int(height),int(width)),np.uint8)
    # face
    points = np.concatenate([landmarks[:17,:],landmarks[17:27,:][::-1,:]],axis=0)
    face_mask = cv2.fillConvexPoly(white_img, points = points, color=(255, 255, 255))
    # mouse & eye
    white_img = np.zeros((int(height),int(width)),np.uint8)
    # mask mouse
    cv2.fillConvexPoly(white_img, points = landmarks[48:60,:], color=(255, 255, 255))
    # mask eye
    cv2.fillConvexPoly(white_img, points = landmarks[36:42,:], color=(255, 255, 255))
    outlier_mask = cv2.fillConvexPoly(white_img, points = landmarks[42:48,:], color=(255, 255, 255))
    # merge mask
    roi_mask = cv2.bitwise_xor(face_mask,outlier_mask)
    return roi_mask

def AveragedRGB(roi):
    """
    RoI領域のRGB信号を平均化して返す
    """
    B_value, G_value, R_value = roi.T
    rgb_component = np.array([[np.mean(R_value), np.mean(G_value),np.mean(B_value)]])
    return rgb_component


def ExportRGBComponents(df,cap,fpath):
    rgb_components = MultipleRoI(df,cap)
    columnslist = []
    for i in range(11):
        columnslist.append("camera{}_B".format(i+1))
        columnslist.append("camera{}_G".format(i+1))
        columnslist.append("camera{}_R".format(i+1))
    df = pd.DataFrame(rgb_components,columns=columnslist)
    df.to_csv(fpath)
    print("########################\n")
    print("ExportData:\n{}".format(fpath))
    print("########################\n")



if __name__ == "__main__":
    vpath = r"C:\Users\akito\source\WebcamRecorder\UmcompressedVideo_3.avi"
    landmark_data = r"C:\Users\akito\source\WebcamRecorder\output\UmcompressedVideo_3.csv"

    #動画の読み込み
    cap = cv2.VideoCapture(vpath)

    #Openfaceで取得したLandMark
    df = pd.read_csv(landmark_data,header = 0).rename(columns=lambda x: x.replace(' ', ''))
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(df.shape)

    data = FaceAreaRoI(df,cap)
    np.savetxt("./result/rgb_ucomp2_faceroi.csv",data,delimiter=",")