"""
自動で実行
"""
# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from biosppy import signals 

# Import local
from ... import config as cf
from ..roi_detection.landmark_extractor import *
from src.pulse_extraction import *
from src.tools.openface import *
from src.tools import visualize
from src.tools.evaluate import *
from src.tools.opensignal import *
from src.tools.peak_detector import *
from src import preprocessing
from src.tools.skintrackbar import *


def ALL_Analysis(input_folder, output_folder):
    file_lists = os.listdir(input_folder)
    for i in range(0,len(file_lists)-2,3):
        pathlist = file_lists[i:i+3]
        CamPath = [s for s in pathlist if 'Cam' in s][0]
        ECGPath = [s for s in pathlist if 'ECG' in s][0]
        TsPath = [s for s in pathlist if 'timestamp' in s][0]
        split_param = CamPath.split()
        print(split_param)

        if any(s.endswith("fps") for s in split_param):
            fps = split_param[3][:-3]
            action_rppg(input_folder,output_folder,CamPath,ECGPath,TsPath,fps=float(fps))
        else:
            action_rppg(input_folder,output_folder,CamPath,ECGPath,TsPath,fps=30)

def action_rppg(indict,outdict,CamPath,ECGPath,TsPath,fps):
    c_fps = 100
    sample_rate = 100 # ECGのサンプリングレート
    

    return 0


def generate_basefile():
    #OPEN FACEを実行
    print(os.path.join(indict,CamPath))
    openface(os.path.join(indict,CamPath), outdict)

    #ROI検出
    LAND_PATH = os.path.join(outdict,CamPath[:-4]+".csv")
    df = pd.read_csv(LAND_PATH, header = 0).rename(columns=lambda x: x.replace(' ', ''))
    SkinPath = CamPath[:-8]+" SkinPram"+".npy"
    rgb_signal = FaceAreaRoI(df, os.path.join(indict,CamPath),os.path.join(outdict,SkinPath))
    np.savetxt(os.path.join(outdict,CamPath[:-8]+" RGB Signals.csv"), rgb_signal, delimiter=",")

    # 信号の目的のレートへのリサンプリング
    data_time = np.loadtxt(os.path.join(indict,TsPath),delimiter=",")
    rgb_signal = preprocessing.rgb_resample(rgb_signal,data_time,fs=fps)
    # RPPG
    rppg_pos = POSMethod(rgb_signal, fs=fps ,filter=True).reshape(-1,1)
    rppg_green = GreenMethod(rgb_signal, fs=fps).reshape(-1,1)
    rppg_softsig = SoftsigMethod(rgb_signal, fs=fps).reshape(-1,1)
    rppg_signals = np.concatenate([rppg_pos,rppg_green,rppg_softsig],axis=1)
    np.savetxt(os.path.join(outdict,CamPath[:-8]+" rPPG Signals.csv"), rppg_signals, delimiter=",")

    # Peakの出力
    est_rpeaks = RppgPeakDetection(rppg_signals[:,0], fs=fps, fr=c_fps, show=False)
    np.savetxt(os.path.join(outdict,CamPath[:-8]+" POS EST RRI.csv"), est_rpeaks, delimiter=",")
    est_rpeaks = RppgPeakDetection(rppg_signals[:,1], fs=fps, fr=c_fps, show=False)
    np.savetxt(os.path.join(outdict,CamPath[:-8]+" Green EST RRI.csv"), est_rpeaks, delimiter=",")
    est_rpeaks = RppgPeakDetection(rppg_signals[:,2], fs=fps, fr=c_fps, show=False)
    np.savetxt(os.path.join(outdict,CamPath[:-8]+" Softsig EST RRI.csv"), est_rpeaks, delimiter=",")

    # リファレンスのピーク出力
    ref_signal = np.loadtxt(os.path.join(indict,ECGPath))
    ref_rpeaks = signals.ecg.ecg(ref_signal, sampling_rate=sample_rate, show=False)[-2]
    np.savetxt(os.path.join(outdict,CamPath[:-8]+" REF RRI.csv"), ref_rpeaks, delimiter=",")

def skinimage():
    cap = cv2.VideoCapture(os.path.join(indict, CamPath))
    SkinPath = CamPath[:-8]+" SkinPram"+".npy"
    ret, frame = cap.read()
    # Import landmark 
    LAND_PATH = os.path.join(outdict,CamPath[:-4]+".csv")
    df = pd.read_csv(LAND_PATH, header = 0).rename(columns=lambda x: x.replace(' ', ''))
    pix_x_frames = df.loc[:, df.columns.str.contains('x_')].values.astype(np.int)
    pix_y_frames = df.loc[:, df.columns.str.contains('y_')].values.astype(np.int)
    pix_x = pix_x_frames[0,:].reshape(-1, 1)
    pix_y = pix_y_frames[0,:].reshape(-1, 1)
    # FaceMask by features point
    mask = RoIDetection(frame,pix_x,pix_y)
    face_img = cv2.bitwise_and(frame, frame, mask=mask)
    skin_mask = sd.SkinDetectTrack(face_img,os.path.join(outdict,SkinPath))
    mask = cv2.bitwise_and(mask, skin_mask, skin_mask)
    mask_img = cv2.bitwise_and(frame, frame, mask=mask)
    outpath = os.path.join(r"D:\rPPGDataset\Figure\Images\shizuya\luminance\skinmask_GUI",
                           CamPath[:-4]+".jpg")
    cv2.imwrite(outpath, mask_img)