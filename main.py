"""
実行ファイル
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
import config as cf
from src.roi_detection.landmark_extractor import *
from src.pulse_extraction import *
from src.tools.openface import *
from src.tools import visualize
from src.tools.evaluate import *
from src.tools.opensignal import *
from src.tools.peak_detector import *
from src import preprocessing
from src.tools.skintrackbar import *

# rpeaks_softsig = np.loadtxt(r"D:\rPPGDataset\Analysis\luminance\tohma\2020-12-29 16-27-46.965417 Front 700lux POS EST RRI.csv")
# rpeaks_ref = np.loadtxt(r"D:\rPPGDataset\Analysis\luminance\tohma\2020-12-29 16-27-46.965417 Front 700lux REF RRI.csv")
# plt.plot(rpeaks_softsig[1:],rpeaks_softsig[1:]-rpeaks_softsig[:-1],label="EST")
# plt.plot(1000*rpeaks_ref[1:],1000*(rpeaks_ref[1:]-rpeaks_ref[:-1]),label="REF")
# plt.legend()
# plt.show()

rppg_sig = np.loadtxt(r"D:\rPPGDataset\Analysis\luminance\tohma\2020-12-29 16-44-19.110293 Front 100lux rPPG Signals.csv",delimiter=",")
ref_rpeaks = np.loadtxt(r"D:\rPPGDataset\Analysis\luminance\tohma\2020-12-29 16-44-19.110293 Front 100lux REF RRI.csv",delimiter=",")
ref_rpeaks *= 1000

# plt.plot(rppg_sig[:,0])
# plt.show()
est_rpeaks = RppgPeakDetection(rppg_sig[:,1], fs=30, fr=100, show=True)

r_est_rpeaks,rri = OutlierDetect(est_rpeaks)
plt.plot(r_est_rpeaks,rri)
plt.show()

exit()
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

    return 0

    
    # LAND_PATH = os.path.join(outdict,CamPath[:-4]+".csv")
    # df = pd.read_csv(LAND_PATH, header = 0).rename(columns=lambda x: x.replace(' ', ''))
    # SKINPATH = os.path.join(outdict,CamPath[:-8]+" SkinPram"+".npy")
    # trackbar(df,os.path.join(indict, CamPath),SKINPATH)
    # return 0

    #OPEN FACEを実行
    # print(os.path.join(indict,CamPath))
    # openface(os.path.join(indict,CamPath), outdict)

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

infolder = r"D:\rPPGDataset\Log\luminance\shizuya"
outfolder = r"D:\rPPGDataset\Analysis\luminance\shizuya"
ALL_Analysis(infolder,outfolder)
# infolder = r"D:\rPPGDataset\Log\framerate\tozyo"
# outfolder = r"D:\rPPGDataset\Analysis\framerate\tozyo"
# ALL_Analysis(infolder,outfolder)
# infolder = r"D:\rPPGDataset\Log\motion\tozyo"
# outfolder = r"D:\rPPGDataset\Analysis\motion\tozyo"
# ALL_Analysis(infolder,outfolder)


exit()








# # rppg_result = np.concatenate([rppg_ts.reshape(-1,1),rppg_pos.reshape(-1,1)],axis=1)
# # np.savetxt(r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\dendai_JointResearch\source\共有データ\rppg_signal.csv",rppg_result,delimiter=",")
# visualize.plot_PSD(est_rpeaks/1000,nfft=2**10)
# plt.show()
# plt.plot(est_rpeaks[1:]/1000,(est_rpeaks[1:]-est_rpeaks[:-1]))
# plt.plot(ref_peaks[1:],1000*(ref_peaks[1:]-ref_peaks[:-1]))
# plt.show()



#reference
peaks_ppg = np.loadtxt(r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\Ikadai\2020-11-24\rpeaks_cest.csv",delimiter=",")
peaks_ref = np.loadtxt(r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\Ikadai\2020-11-24\rpeaks_cref.csv",delimiter=",")

print(len(peaks_ref))
print(len(peaks_ppg))
rri_ref = peaks_ref[1:]-peaks_ref[:-1]
rri_est = peaks_ppg[1:]-peaks_ppg[:-1]
# plt.scatter(rri_ref[:-2],rri_est[1:])
print(np.corrcoef(rri_ref,rri_est))

# plt.plot(peaks_ref[1:]/60000,peaks_ref[1:]-peaks_ref[:-1],label="Reference")
# plt.plot(peaks_ppg[1:]/60000,peaks_ppg[1:]-peaks_ppg[:-1],label="Estimation")

# visualize.plot_BlandAltman(peaks_ref/1000,peaks_ppg/1000)
# plt.show()
# visualize.plot_spectrograms(rppg_IC,fs=50,tw=10)

# df = biosignal_time_summary(peaks_ref, duration=180,overlap=90)
# df.to_excel(r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\Ikadai\2020-11-24\hrv_analysis_ref2.xlsx")

# exit()
# est_rpeaks = RppgPeakDetection(rppg_IC,time_IC, fr=50,show=True, filter=True, col=0.01)

# # plt.plot(est_rpeaks[1:],est_rpeaks[1:]-est_rpeaks[:-1])
# # plt.plot(ref_rpeaks[1:],ref_rpeaks[1:]-ref_rpeaks[:-1])
# plt.plot(ref_rpeaks[1:]-ref_rpeaks[:-1],est_rpeaks[1:]-est_rpeaks[:-1])
# plt.show()


# # ファイル名からフレーム時刻を取得
# timestamps = []
# for file in files:
#     file_param = re.split('[_ ]', file)
#     timestamp = re.split('[-]', file_param[1])
#     # Str to int
#     timestamp = [float(s) for s in timestamp]
#     timestamps.append(timestamp[0]*60**2 + timestamp[1]*60 + timestamp[2])

# data_time = np.array(timestamps)
# data_time = data_time - data_time[0]
# data_timediff = 1/np.diff(data_time)
# print(np.mean(data_timediff))

data_time = np.loadtxt(r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\Ikadai\2020-11-24\timestamp.csv",
                        delimiter=",")

# RPPG
rgb_signal = np.loadtxt(cf.OUTPUT_PATH, delimiter=",")

# 標準化
_, r_resamp = linear_resampling(rgb_signal[:,0],data_time, fr=40)
_, g_resamp = linear_resampling(rgb_signal[:,1],data_time, fr=40)
data_time, b_resamp = linear_resampling(rgb_signal[:,2],data_time, fr=40)
rgb_signal = np.concatenate([r_resamp.reshape(-1,1),
                             g_resamp.reshape(-1,1),
                             b_resamp.reshape(-1,1)],
                             axis=1)

rppg_pos = POSMethod(rgb_signal, fs=40, filter=True)
# rppg_green = GreenMethod(rgb_signal,fs=40)
# rppg_chrom = ChromMethod(rgb_signal, fs=40)

_,axes = plt.subplots(3,1,sharex=True)
axes[0].plot(data_time,rppg_pos)

data = np.concatenate([data_time.reshape(-1,1),rppg_pos.reshape(-1,1)],axis=1)
np.savetxt(r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\Ikadai\2020-11-24\ippg_signal.csv",data,delimiter=",")
ref_ppg = np.loadtxt(cf.REF_PATH)[:,-1]
np.savetxt(r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\Ikadai\2020-11-24\ppg_signal.csv",ref_ppg,delimiter=",")
exit()



# axes[1].plot(data_time,rppg_green)

# rpeaks_ppg = RppgPeakDetection(rppg_pos,
#                                data_time,
#                                fr=100,
#                                show=True,
#                                filter=False,
#                                col=0.20)

# # #reference
# # ref_ppg = np.loadtxt(cf.REF_PATH)[:,-1]#[int(100*cf.DELAY_T):int(100*cf.DURATION_T),-1]
# # rpeaks_ref = RppgPeakDetection(ref_ppg,
# #                                np.arange(0, len(ref_ppg)/1000, 1/1000),
# #                                fr=1000,
# #                                show=False,
# #                                filter=True,
# #                                col=0.20)
# # np.savetxt(r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\Ikadai\2020-11-24\rpeaks_ref.csv",rpeaks_ref,delimiter=",")
# np.savetxt(r"C:\Users\akito\Desktop\Hassylab\projects\RPPG\Ikadai\2020-11-24\rpeaks_ppg2.csv",rpeaks_ppg,delimiter=",")
# plt.show()


# # RPPG
# fps_IMX290=52
# fps_IC = 50

# rgb_IMX290 = np.loadtxt(r"C:\Users\akito\Desktop\20201111\rgb_20201111_imx290.csv",
#                         delimiter=",")[int(fps_IMX290*(cf.DELAY_T+0.2)):int(fps_IMX290*cf.DURATION_T),:]

# rgb_IC = np.loadtxt(r"C:\Users\akito\Desktop\20201111\rgb_20201111_dkf.csv",
#                         delimiter=",")[int(fps_IC*cf.DELAY_T):int(fps_IC*cf.DURATION_T),:]
# #reference
# ref_ppg = np.loadtxt(cf.REF_PATH)[int(100*cf.DELAY_T):int(100*cf.DURATION_T),-1]

# rppg_IMX290 = GreenMethod(rgb_IMX290, fs=fps_IMX290)
# rppg_IC = GreenMethod(rgb_IC, fs=fps_IC)

# rpeaks_IC = RppgPeakDetection(rppg_IC,
#                               data_time[int(fps_IC*cf.DELAY_T):int(fps_IC*cf.DURATION_T)]-cf.DELAY_T,
#                               fr=100,show=False, filter=False, col=0.15)
# # np.savetxt(r"C:\Users\akito\Desktop\20201111\rpeaks_IC.csv",
# #            rpeaks_IC,delimiter=",")
# rpeaks_IMX290 = RppgPeakDetection(rppg_IMX290,
#                                np.arange(0, len(rppg_IMX290)/fps_IMX290, 1/fps_IMX290),
#                                fr=100,show=False, filter=False, col=0.20)
# # np.savetxt(r"C:\Users\akito\Desktop\20201111\rpeaks_IMX290.csv",
# #            rpeaks_IMX290,delimiter=",")



# plt.figure()
# visualize.plot_PSD(rri_peaks=rpeaks_IC/1000, label="DFK23U618")
# plt.show()

# visualize.plot_PSD(rri_peaks=rpeaks_ppg/1000, label="PPG")

# plt.show()
# visualize.plot_PSD(rri_peaks=rpeaks_IMX290/1000, label="Sony IMX290")

# plt.show()

# import pyhrv
# pyhrv.frequency_domain.welch_psd(nni =rpeaks_IC[1:] - rpeaks_IC[:-1],show=True,detrend=False,nfft=2**9)
# pyhrv.frequency_domain.welch_psd(nni =rpeaks_ppg[1:] - rpeaks_ppg[:-1],show=True,detrend=False,nfft=2**9)
# pyhrv.frequency_domain.welch_psd(nni =rpeaks_IMX290[1:] - rpeaks_IMX290[:-1],show=True,detrend=False,nfft=2**9)
# np.savetxt(r"C:\Users\akito\Desktop\20201111\rpeaks_ppg.csv",
#            rpeaks_ppg,delimiter=",")


# visualize.plot_snr(rppg_IMX290, fs=fps_IMX290,text="Sony IMX290")
# visualize.plot_snr(rppg_IC, fs=fps_IC,text="DFK23U618")
# ref_ppg = preprocessing.ButterFilter(ref_ppg, 0.7, 2.5, fs=100)
# visualize.plot_snr(ref_ppg, fs=100,text="PPG")


# _,axes = plt.subplots(3,1,sharex=True)
# ts_IMX = np.arange(0, len(rppg_IMX290)/fps_IMX290, 1/fps_IMX290)
# axes[0].plot(ts_IMX,rppg_IMX290)
# axes[0].set_title("Sony IMX290")
# ts_IC = np.arange(0, len(rppg_IC)/fps_IC, 1/fps_IC)
# axes[1].plot(data_time[int(fps_IC*cf.DELAY_T):int(fps_IC*cf.DURATION_T)]-cf.DELAY_T,rppg_IC)
# axes[1].set_title("DFK23U618")
# ts_PPG = np.arange(0, len(ref_ppg)/100, 1/100)
# axes[2].plot(ts_PPG,ref_ppg)
# axes[2].set_title("PPG")
# plt.legend()

# visualize.plot_psdgraph(rgb_IMX290-np.mean(rgb_IMX290), fs=52,title="IMX290")
# visualize.plot_psdgraph(rgb_IC-np.mean(rgb_IC), fs=50,title="IC")
plt.show()
exit()
# rppg_IMX290 = preprocessing.ButterFilter(rppg_IMX290,0s.7, 2.5, fs=52)
# time_IMX290 = np.arange(0, len(rppg_IMX290)/52, 1./52)
# rppg_IC = preprocessing.ButterFilter(rppg_IC,0.7, 2.5, fs=50)
# time_IC = np.arange(0, len(rppg_IC)/50, 1./50)
# visualize.plot_snr(rppg_IC[:1500], hr=None, fs=50)
# visualize.plot_snr(rppg_IMX290, hr=None, fs=52)


# _,axes= plt.subplots(2,1,sharex=True)
# [time_IMX290<=80]
# axes[0].set_title("Sony IMX290")
# axes[0].plot(time_IMX290[time_IMX290<=80],
#              rppg_IMX290[time_IMX290<=80])
# axes[1].set_title("IC DFK23U619")
# axes[1].plot(time_IC, rppg_IC)
# plt.show()
# exit()



# ref_peaks = signals.ecg.ecg(ref_ecg, sampling_rate=1000, show=True)[2]
# ts = np.arange(0, len(ref_ecg)*0.001, 0.001)


# visualize.plot_snr(rppg_IMX290, hr=None, fs=52)

# visualize.plot_snr(rppg_IC, hr=None, fs=50)
# plt.show()


# Cut-Time
# rppg_pos = rppg_pos[(data_time>cf.DELAY_T) & (data_time<cf.DURATION_T+cf.DELAY_T)]
# data_time = data_time[(data_time>cf.DELAY_T) & (data_time<cf.DURATION_T+cf.DELAY_T)]-cf.DELAY_T

# # Reference
# ref_ecg = np.loadtxt(cf.REF_PATH)[int(cf.DELAY_T*1000):int((cf.DURATION_T+cf.DELAY_T)*1000), -3]
# ref_peaks = signals.ecg.ecg(ref_ecg, sampling_rate=1000, show=True)[2]
# ts = np.arange(0, len(ref_ecg)*0.001, 0.001)
# plt.plot(ref_peaks[1:], ref_peaks[1:]-ref_peaks[:-1],label="REF")
# plt.plot(est_rpeaks[1:], est_rpeaks[1:]-est_rpeaks[:-1],label="EST")
# plt.legend()
# plt.show()

# ref_peaks = ref_peaks[:len(est_rpeaks)]
# print(len(est_rpeaks))
# print(len(ref_peaks))
# visualize.plot_BlandAltman(est_rpeaks*0.001,ref_peaks*0.001)


# result = np.concatenate((ref_peaks.reshape(-1,1),
#                          est_rpeaks.reshape(-1,1)), axis=1)

# np.savetxt(cf.RRI_OUTPATH,result,delimiter=",")


# # #plt.legend()
# import pyhrv
# pyhrv.frequency_domain.welch_psd(nni = est_rpeaks[1:] - est_rpeaks[:-1],show=True,detrend=False,nfft=2**9)
# pyhrv.frequency_domain.welch_psd(nni = ref_peaks[1:] - ref_peaks[:-1], show=True,detrend=False,nfft=2**9)
# visualize.plot_PSD(est_rpeaks*0.001, label="EST")
# visualize.plot_PSD(ref_peaks*0.001, label="REF")
# plt.show()

# #_,axes = plt.subplots(2,1,sharex=True)
# #axes[0].plot(data_time[data_time>cf.DELAY_T]-cf.DELAY_T,rppg_pos[data_time>cf.DELAY_T])
# #axes[1].plot(ts, ref_ecg)
# plt.show()

# cap = cv2.VideoCapture(cf.vpath)
# # # Openfaceで取得したLandMark
# df = pd.read_csv(cf.landmark_data, header = 0).rename(columns=lambda x: x.replace(' ', ''))

# print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(df.shape)
# ppg_signal = FaceAreaRoI(df,cap)
# np.savetxt(outpath, ppg_signal, delimiter=",")
# ref_ecg = np.loadtxt(refpath)[int(delay*1000):, -1]
# ref_peaks = signals.ecg.ecg(ref_ecg, sampling_rate=1000, show=False)[2]

# rgb_signal = np.loadtxt(outpath, delimiter=",")
# rppg_pos = POSMethod(rgb_signal, WinSec=1.6, fs=30, filter=False)
# rppg_green = GreenMethod(rgb_signal, fs=30)
# result = np.concatenate((rppg_pos.reshape(-1,1),rppg_green.reshape(-1,1)), axis=1)
# np.savetxt(r"C:\Users\akito\Desktop\rppg_motion_talking.csv",result,delimiter=",")
# _, axes= plt.subplots(2,1,sharex=True)
# axes[0].plot(rppg_pos)
# axes[1].plot(rppg_green)
# plt.show()

# # estimate
# t_T, HR_T = CalcTimeHR(rpeaks, rri, segment=17.06)
# t_F, HR_F = CalcFreqHR(filter_rppg)
# # reference
# ref_t_T, ref_HR_T = CalcTimeHR(ref_rpeaks, ref_rri, segment=17.06)
# ref_t_F, ref_HR_F = CalcFreqHR(ref_ppg)

# #plt.plot(ref_t_F, ref_HR_F, label="PPG FreqHR")
# plt.plot(ref_t_T, ref_HR_T, label="PPG TimeHR")
# plt.plot(t_F, HR_F, label="RPPG FreqHR")
# plt.plot(t_T, HR_T, label="RPPG TimeHR")
# plt.xlabel("Time[s]")
# plt.ylabel("HR [bpm]")

# plt.legend()

# print("MAE: {}".format(np.mean(abs(ref_HR_F-HR_F))))
# print("RMSE: {}".format(np.sqrt(np.mean((ref_HR_F-HR_F)**2))))
# print("Cov: {}".format(np.corrcoef(ref_HR_F,HR_F)[0, 1]))
# plt.show()