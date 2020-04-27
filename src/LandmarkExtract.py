# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import cv2

# ActionUnit制定
def ActionUnit():
    #表情の枠を作成
    surprize_array = np.array([])
    fear_array = np.array([])
    disgust_array = np.array([])
    anger_array = np.array([])
    happiness_array = np.array([])
    sadness_array = np.array([])
    AU_1 =float(df.AU01_r[i])
    AU_2 =float(df.AU02_r[i])
    AU_4 =float(df.AU04_r[i])
    AU_5 =float(df.AU05_r[i])
    AU_6 =float(df.AU06_r[i])
    AU_7 =float(df.AU07_r[i])
    AU_9 =float(df.AU09_r[i])
    AU_10 =float(df.AU10_r[i])
    AU_12 =float(df.AU12_r[i])
    AU_14 =float(df.AU14_r[i])
    AU_15 =float(df.AU15_r[i])
    AU_17 =float(df.AU17_r[i])
    AU_20 =float(df.AU20_r[i])
    AU_23 =float(df.AU23_r[i])
    AU_25 =float(df.AU25_r[i])
    AU_26 =float(df.AU26_r[i])
    
    #アクションユニットから各表情のスコアを算出
    surprize = float((AU_1*40 + AU_2*30 + AU_5*60 + AU_15*20
                     + AU_20*10 + AU_26*60)/(40+30+60+20+10+60))
    fear = float((AU_1*50 + AU_2*10 + AU_4*80 + AU_5*60
                     + AU_15*30 + AU_20*10 + AU_26*30)/(50+10+80+60+30+10+30))
    disgust = float((AU_2*60 + AU_4*40 + AU_9*20
                      + AU_15*60 + AU_17*30)/(60+40+20+60+30))
    anger = float((AU_2*30 + AU_4*60 + AU_7*50 + AU_9*20
                     + AU_10*10 + AU_20*15 + AU_26*30)/(30+60+50+20+10+15+30))
    happiness = float((AU_1*65 + AU_6*70 + AU_12*10 + AU_14*10)/(65+70+10+10))

    sadness = float((AU_1*40 + AU_4*50 + AU_15*40 + AU_23*20)/(40+50+40+20))

    pass

def GetRGBFromRoI(roi):
    B_value, G_value, R_value = roi.T
    rgb_component = np.array([[np.mean(B_value), np.mean(G_value),np.mean(R_value)]])
    return rgb_component

# 顔のサイズを初期化する
def ScanFaceSize(df,initframe=450):
    a = 0.90
    wide_face      = int((df.x_29[initframe]-df.x_2[initframe])*a)
    wide_nose      = int((df.x_29[initframe]-df.x_39[initframe])*a)
    hight_eye      = int((df.y_29[initframe]-df.y_40[initframe])*a)
    hight_cheek    = int((df.y_33[initframe]-df.y_29[initframe])*a)
    hight_Eyebrows = int((df.y_29[initframe]-df.y_27[initframe])*a)
    hignht_nose    = int((df.y_30[initframe]-df.y_29[initframe])*a)

    return wide_face,wide_nose,hight_eye,hight_cheek,hight_Eyebrows,hignht_nose

def SelectRoI(df,cap):
    # 顔スキャンし初期化
    wide_face, wide_nose, hight_eye, hight_cheek, hight_Eyebrows, hignht_nose = ScanFaceSize(df)
    i = 0
    pix_x_frames = df.iloc[:,:69].values.astype(np.int)
    pix_y_frames = np.floor(df.iloc[:,-69:].values).astype(np.int)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter を作成する。
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    writer = cv2.VideoWriter("20200426_test.avi", fourcc, fps, (width, height))


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
        cv2.rectangle(frame, (pix_x[31], pix_y[29] + hight_cheek), (pix_x[31] - wide_face, pix_y[29]), color=(0,0,255),thickness= 4)
        cv2.rectangle(frame, (pix_x[29] + wide_nose, pix_y[29]), (pix_x[29] - wide_nose, pix_y[28]), color=(0,0,255),thickness= 4)
        for roi in roi_list:
            rgb_component = GetRGBFromRoI(roi)
            rgb_item = np.concatenate([rgb_item,rgb_component],axis=1)
        

        # マージ処理
        if allRGBArrays is None:
            allRGBArrays = rgb_item
        else:
            allRGBArrays = np.concatenate([allRGBArrays,rgb_item],axis=0)
        
        writer.write(frame)  # フレームを書き込む。


    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    return allRGBArrays


def ExportRGBComponents(df,cap,fpath):
    rgb_components = SelectRoI(df,cap)
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
