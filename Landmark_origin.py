# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 20:32:35 2020

@author: yuya4
"""

#外部ライブラリのインポート
import numpy as np
import cv2
import pandas as pd
import time
#from PIL import Image, ImageFilter

#動画の読み込み
cap = cv2.VideoCapture(r"C:\Users\akito\source\WebcamRecorder\UmcompressedVideo_origin.avi")

#i初期化
i=0
#解析時間の測定(初期時間の取得)
start=time.time()
#動画総フレーム数の取得
count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#抽出領域枠の設定
array = np.array([])

green_array1 = np.array([])
green_array2 = np.array([])
green_array3 = np.array([])
green_array4 = np.array([])
green_array5 = np.array([])
green_array6 = np.array([])
green_array7 = np.array([])
green_array8 = np.array([])
green_array9 = np.array([])
green_array10 = np.array([])
green_array11 = np.array([])

#表情の枠を作成

surprize_array = np.array([])
fear_array = np.array([])
disgust_array = np.array([])
anger_array = np.array([])
happiness_array = np.array([])
sadness_array = np.array([])
#挿入するフレームを作成

#framenumber = np.array([frame,1,2,3])
df_out = pd.DataFrame(columns= ['frame','camera1','camera2','camera3','camera4','camera5',
                                'camera6','camera7','camera8','camera9','camera10','camera11'])

df_emotion = pd.DataFrame(columns= ['surprize','fear','disgust','anger'
                                    ,'happiness','sadness'])

#Openfaceで取得した顔の座標点をcsvから読み込み
df = pd.read_csv(r"C:\Users\akito\source\WebcamRecorder\output\UmcompressedVideo_origin.csv")

df = df.rename(columns=lambda x: x.replace(' ', ''))
#顔のランドマークを用いた座標定義

#x0 = int(df.x_0[0])#左頬
#x2 = int(df.x_2[0])#左頬
#x4 = int(df.x_4[0])
#x6 = int(df.x_6[0])
#x8 = int(df.x_8[0])#顎の中心
#x10 = int(df.x_10[0])
#x12 = int(df.x_12[0])
#x14 = int(df.x_14[0])#右頬
#x16 = int(df.x_16[0])#左頬
#
#x27 = int(df.x_27[0])#眉間
#x28 = int(df.x_28[0])
#x29 = int(df.x_29[0])
#x30 = int(df.x_30[0])#鼻頭
#
#x31 = int(df.x_31[0])#鼻の左下
#x33 = int(df.x_33[0])#鼻の中央
#x35 = int(df.x_35[0])#鼻の右下
#
#x48 = int(df.x_48[0])#左唇端
#x51 = int(df.x_51[0])#上唇頭
#x54 = int(df.x_54[0])#右唇端
#x57 = int(df.x_57[0])#下唇頭
#
#x36 = int(df.x_36[0])#左目左端
#x39 = int(df.x_39[0])#左目右端
#x42 = int(df.x_42[0])#
#x45 = int(df.x_45[0])#下唇頭
#
#y0 = int(df.y_0[0])#左頬
#y2 = int(df.y_2[0])#左頬
#y4 = int(df.y_4[0])
#y6 = int(df.y_6[0])
#y8 = int(df.y_8[0])#顎の中心
#y10 = int(df.y_10[0])
#y12 = int(df.y_12[0])
#y14 = int(df.y_14[0])#右頬
#y16 = int(df.y_16[0])#左頬
#
#y27 = int(df.y_27[0])#眉間
#y28 = int(df.y_28[0])
#y29 = int(df.y_29[0])
#y30 = int(df.y_30[0])#鼻頭
#
#y31 = int(df.y_31[0])#鼻の左下
#y33 = int(df.y_33[0])#鼻の中央
#y35 = int(df.y_35[0])#鼻の右下
#
#y48 = int(df.y_48[0])#左唇端
#y51 = int(df.y_51[0])#上唇頭
#y54 = int(df.y_54[0])#右唇端
#y57 = int(df.y_57[0])#下唇頭
#
#y36 = int(df.y_36[0])#左目左端
#y39 = int(df.y_39[0])#左目右端
#y42 = int(df.y_42[0])#
#y45 = int(df.y_45[0])#下唇頭



#顔の回転角度の読み込み
#roll = int(df.x_45[0])
#yaw = int(df.x_45[0])
#pitch = int(df.x_45[0])

# 顔の幅、顔の高さの値を取得

wide_face = int((df.x_29[450]-df.x_2[450])*9/10)

wide_nose = int((df.x_29[450]-df.x_39[450])*9/10)

hight_eye = int((df.y_29[450]-df.y_40[450])*9/10)

hight_cheek = int((df.y_33[450]-df.y_29[450])*9/10)

hight_Eyebrows = int((df.y_29[450]-df.y_27[450])*9/10)

hignht_nose = int((df.y_30[450]-df.y_29[450])*9/10)



for i in range(count):
    print('frame:', i)
    ret, frame = cap.read()
    if ret==False:
        print("読み取り失敗")
        break

    x0 = int(df.x_0[i])#左頬
    x2 = int(df.x_2[i])#左頬
    x4 = int(df.x_4[i])
    x6 = int(df.x_6[i])
    x8 = int(df.x_8[i])#顎の中心
    x10 = int(df.x_10[i])
    x12 = int(df.x_12[i])
    x14 = int(df.x_14[i])#右頬
    x16 = int(df.x_16[i])#左頬

    x27 = int(df.x_27[i])#眉間
    x28 = int(df.x_28[i])
    x29 = int(df.x_29[i])
    x30 = int(df.x_30[i])#鼻頭

    x31 = int(df.x_31[i])#鼻の左下
    x33 = int(df.x_33[i])#鼻の中央
    x35 = int(df.x_35[i])#鼻の右下

    x48 = int(df.x_48[i])#左唇端
    x51 = int(df.x_51[i])#上唇頭
    x54 = int(df.x_54[i])#右唇端
    x57 = int(df.x_57[i])#下唇頭

    x36 = int(df.x_36[i])#左目左端
    x39 = int(df.x_39[i])#左目右端
    x42 = int(df.x_42[i])#
    x45 = int(df.x_45[i])#下唇頭

    y0 = int(df.y_0[i])#左頬
    y2 = int(df.y_2[i])#左頬
    y4 = int(df.y_4[i])
    y6 = int(df.y_6[i])
    y8 = int(df.y_8[i])#顎の中心
    y10 = int(df.y_10[i])
    y12 = int(df.y_12[i])
    y14 = int(df.y_14[i])#右頬
    y16 = int(df.y_16[i])#左頬

    y27 = int(df.y_27[i])#眉間
    y28 = int(df.y_28[i])
    y29 = int(df.y_29[i])
    y30 = int(df.y_30[i])#鼻頭

    y31 = int(df.y_31[i])#鼻の左下
    y33 = int(df.y_33[i])#鼻の中央
    y35 = int(df.y_35[i])#鼻の右下

    y48 = int(df.y_48[i])#左唇端
    y51 = int(df.y_51[i])#上唇頭
    y54 = int(df.y_54[i])#右唇端
    y57 = int(df.y_57[i])#下唇頭

    y36 = int(df.y_36[i])#左目左端
    y39 = int(df.y_39[i])#左目右端
    y42 = int(df.y_42[i])#
    y45 = int(df.y_45[i])#下唇頭

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


#顔の移動を鼻の頭を基準に計算

        # xy座標の順番注意
    y_list = [y27, y28, y29, y30, y31]
    y_list = sorted(y_list)
    y_list = np.clip(y_list, 0, None)
    new_y_list=[]
    for idx in range(len(y_list)-1):
        if y_list[idx]>=y_list[idx+1]:
            y_list[idx+1]=y_list[idx]+1
        new_y_list.append(y_list[idx])

    new_y_list.append(y_list[-1])
    y27, y28, y29, y30, y31 = new_y_list
    roi_1 =frame[y29:y29 + hight_cheek, x31 - wide_face:x31]#右頬
    roi_2 =frame[y29:y29 + hight_cheek, x35:x35 + wide_face ]#左頬
    roi_3 =frame[y27:y30, x29 - wide_nose:x29 + wide_nose]#鼻全体
    roi_4 =frame[y27:y28, x29 - wide_nose:x29 + wide_nose]#鼻上
    roi_5 =frame[y28:y29, x29 - wide_nose:x29 + wide_nose]#鼻真ん中
    roi_6 =frame[y29:y30, x29 - wide_nose:x29 + wide_nose]#鼻下
    roi_7 =frame[np.clip(y29 - hight_eye, 0, None):y29 + hight_cheek, x31 - wide_face:x31]#右頬ワイド
    roi_8 =frame[np.clip(y29 - hight_eye, 0, None):y29 + hight_cheek, x35:x35 + wide_face ]#左頬ワイド
    roi_9 =frame[y29:y31, x29 - wide_face:x29 + wide_face]#全体
    roi_10 =frame[np.clip(y29 - hight_eye, 0, None):y31, x29 - wide_face:x29 + wide_face]#全体ワイド
    roi_11 =frame[y29:y30, x29 - wide_face:x29 + wide_face]#全体スモール


    cv2.rectangle(frame, (x31, y29 + hight_cheek), 
                         (x31 - wide_face, y29), color=(0,0,255),thickness= 4)
    cv2.imshow('frame',frame)

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


#各領域のRGBチャンネルを分解、green成分を抽出
    rgb_1=cv2.split(roi_1)
    rgb_2=cv2.split(roi_2)
    rgb_3=cv2.split(roi_3)
    rgb_4=cv2.split(roi_4)
    rgb_5=cv2.split(roi_5)
    rgb_6=cv2.split(roi_6)
    rgb_7=cv2.split(roi_7)
    rgb_8=cv2.split(roi_8)
    rgb_9=cv2.split(roi_9)
    rgb_10=cv2.split(roi_10)
    rgb_11=cv2.split(roi_11)

#G成分の値をリストに追加 領域の大きさによって重みづけしないといけないからあとで書き換え
    green_1=np.average(rgb_1[1])
    green_2=np.average(rgb_2[1])
    green_3=np.average(rgb_3[1])
    green_4=np.average(rgb_4[1])
    green_5=np.average(rgb_5[1])
    green_6=np.average(rgb_6[1])
    green_7=np.average(rgb_7[1])
    green_8=np.average(rgb_8[1])
    green_9=np.average(rgb_9[1])
    green_10=np.average(rgb_10[1])
    green_11=np.average(rgb_11[1])

#抽出領域の枠線表示

    array1 = [green_1]
    array2 = [green_2]
    array3 = [green_3]
    array4 = [green_4]
    array5 = [green_5]
    array6 = [green_6]
    array7 = [green_7]
    array8 = [green_8]
    array9 = [green_9]
    array10 = [green_10]
    array11 = [green_11]

    array_ave1 = np.average(array1)
    array_ave2 = np.average(array2)
    array_ave3 = np.average(array3)
    array_ave4 = np.average(array4)
    array_ave5 = np.average(array5)
    array_ave6 = np.average(array6)
    array_ave7 = np.average(array7)
    array_ave8 = np.average(array8)
    array_ave9 = np.average(array9)
    array_ave10 = np.average(array10)
    array_ave11 = np.average(array11)

    green_array1 = np.append(green_array1,array_ave1)
    green_array2 = np.append(green_array2,array_ave2)
    green_array3 = np.append(green_array3,array_ave3)
    green_array4 = np.append(green_array4,array_ave4)
    green_array5 = np.append(green_array5,array_ave5)
    green_array6 = np.append(green_array6,array_ave6)
    green_array7 = np.append(green_array7,array_ave7)
    green_array8 = np.append(green_array8,array_ave8)
    green_array9 = np.append(green_array9,array_ave9)
    green_array10 = np.append(green_array10,array_ave10)
    green_array11 = np.append(green_array11,array_ave11)

    # 表情の値を追加
    surprize_array = np.append(surprize_array,surprize)
    fear_array = np.append(fear_array,fear)
    disgust_array = np.append(disgust_array,disgust)
    anger_array = np.append(anger_array,anger)
    happiness_array = np.append(happiness_array,happiness)
    sadness_array = np.append(sadness_array,sadness)

#    if ret==True:
#        cv2.imshow('video image', roi_4)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break

    i+=1
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
#結果の書き出し
df_out['camera1'] = green_array1
df_out['camera2'] = green_array2
df_out['camera3'] = green_array3
df_out['camera4'] = green_array4
df_out['camera5'] = green_array5
df_out['camera6'] = green_array6
df_out['camera7'] = green_array7
df_out['camera8'] = green_array8
df_out['camera9'] = green_array9
df_out['camera10'] = green_array10
df_out['camera11'] = green_array11

df_out.to_csv('1_kaneko_ROI222.csv',index=False)

df_emotion['surprize'] = surprize_array
df_emotion['fear'] = fear_array
df_emotion['disgust'] = disgust_array
df_emotion['anger'] = anger_array
df_emotion['happiness'] = happiness_array
df_emotion['sadness'] = sadness_array

df_emotion.to_csv('1_kaneko_emotion222.csv',index=False)


#計測時間の表示
e_time=time.time()
print("処理時間="+str(e_time-start))
