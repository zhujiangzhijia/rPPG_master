"""
評価指標を算出する
"""
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal,interpolate
import pyhrv



# 一定時間ごとの特徴量を算出し，dataframe型にまとめて返す
def biosignal_time_summary(rpeaks, duration=300,overlap=150,skip_time=None,outpath=None): 
    # 時間変数を作成
    time_ = np.arange(duration, rpeaks.max()/1000, overlap)
    label_ = time_
    if skip_time is not None:
        time_ = time_ + skip_time
    section_ = zip((time_ - duration), time_)
    emotion = dict(zip(label_.tolist(), section_))

    df = pd.DataFrame([])
    # 各生体データを時間区間りで算出
    for i,key in enumerate(emotion.keys()):
        # セグメント内での特徴量算出
        segment_bio_report = {}
        # 心拍をセクションごとに分割する
        seg_rpeaks  = rpeaks[(rpeaks>=emotion[key][0]*1000) & (rpeaks<=emotion[key][1]*1000)]
        # 特徴抽出
        # bio_parameter = Calc_PSD(seg_rpeaks)
        features = pyhrv.frequency_domain.welch_psd(nni=seg_rpeaks[1:] - seg_rpeaks[:-1],show=False,detrend=False,nfft=2**9)
        # Print all the parameters keys and values individually
        bio_parameter = {"LF_ABS":features["fft_abs"][0],"HF_ABS":features["fft_abs"][1], "LFHFratio": features["fft_ratio"]}
        
        print("{}... done".format(key))
        segment_bio_report.update({'section':key})
        segment_bio_report.update(bio_parameter)

        if i == 0:
            df = pd.DataFrame([], columns=segment_bio_report.keys())

        df =  pd.concat([df, pd.DataFrame(segment_bio_report , index=[key])])
    return df


def CalcSNR(ppg, HR_F=None, fs=30, nfft=512):
    """
    CHROM参照
    SNRを算出する
    SNR が大きいと脈拍の影響が大きく，
    SNRが小さいとノイズの影響が大きい
    """
    freq, power = signal.welch(ppg, fs, nfft=nfft, detrend="constant",
                                     scaling="spectrum", window="hamming")
    # peak hr
    if HR_F is None:
        HR_F = freq[np.argmax(power)]

    # 0.2Hz帯
    GTMask1 = (freq >= HR_F-0.1) & (freq <= HR_F+0.1)
    GTMask2 = (freq >= HR_F*2-0.2) & (freq <= HR_F*2+0.2)
    SPower = np.sum(power[GTMask1 | GTMask2])
    FMask2 = (freq >= 0.5)&(freq <= 4)
    AllPower = np.sum(power[FMask2])
    SNR = 10*np.log10((SPower)**2/(AllPower-SPower)**2)
    return {"HR":HR_F,"SNR":SNR}


def CalcFreqHR(ppg, fs=30, nfft=512):
    """
    Calculate Frequency domain heart rate
    using DFT,
    return HR[bpm]
    """
    # FFT PSD
    f, t, Sxx = signal.spectrogram(ppg, fs, nperseg=nfft,
                                   noverlap=nfft/2, scaling="spectrum")
    # Calc HR
    HR_F = np.array([[]])
    for i in range(len(t)):
        Sxx_t = Sxx[:, i]
        HR_F_t = 60*f[np.argmax(Sxx_t)]
        HR_F = np.append(HR_F, HR_F_t)
    return t, HR_F


def CalcTimeHR(rpeaks, rri, segment=17.06, overlap=None):
    """
    Time domain Heart rate
    """
    if overlap is None:
        overlap = segment/2 
    starts = np.arange(0, rpeaks[-1]-overlap, overlap)
    HR_T = np.array([[]])
    for start in starts:
        end = start + segment
        item_rri = rri[(rpeaks >= start) & (rpeaks < end)]
        ave_hr = 60/np.average(item_rri)
        HR_T = np.append(HR_T, ave_hr)
    ts = starts + overlap
    return ts, HR_T 

def CalcEvalRRI(ref_rri, est_rri):
    corr = np.corrcoef(est_rri, ref_rri)[0, 1]
    x = 0.5*(est_rri + ref_rri)
    y = (est_rri - ref_rri)
    mae = np.mean(abs(y))
    rmse = np.sqrt(np.mean(y**2))
    result = {"corr":corr,"mae":mae,"rmse":rmse}
    return result



def Calc_PSD(rri_peaks, rri=None, nfft=2**8):
    """
    PSDを出力
    rri_peaks [s]
    """
    sample_rate = 4
    if rri is None:
        rri = np.diff(rri_peaks)
        rri_peaks = rri_peaks[1:] - rri_peaks[1]

    # 3次のスプライン補間
    rri_spline = interpolate.interp1d(rri_peaks, rri, 'cubic')
    t_interpol = np.arange(rri_peaks[0], rri_peaks[-1], 1./sample_rate)
    rri_interpol = rri_spline(t_interpol)
    frequencies, powers  = signal.welch(x=rri_interpol, fs=sample_rate, window='hamming',
                                        detrend="constant",	nperseg=len(rri_interpol),
                                        nfft=len(rri_interpol), scaling='density')
    LF = np.sum(powers[(frequencies>=0.05) & (frequencies<0.15)]) * 0.10
    HF = np.sum(powers[(frequencies>0.15) & (frequencies<=0.40)]) * 0.25
    print("Result :LF={:2f}, HF={:2f}, LF/HF={:2f}".format(LF, HF, LF/HF))
    return {"LF_abs":LF,"HF_abs":HF,"LFHFratio":LF/HF}



def Calc_MissPeaks(est_rpeaks=None, 
                   ref_rpeaks=None,
                   threshold=0.25):
    """
    リファレンスのピークと，推定したピーク値を比較
    ピーク検出に失敗する割合を算出する
    ----------
	ref_rpeaks, est_rpeaks : array
		R-peak locations in [ms]
    threshold: float
        外れ値を検出するレベル．
        この値が高いほど外れ値と検出されるピークは多くなる
    ----------
    threshold level
    
    very low : 0.45sec
    low : 0.35sec
    medium : 0.25sec
    strong : 0.15sec
    very strong : .05sec
    """
    
    # ピーク時間のずれを補正
    # 最初のピーク位置でECGとPPGの位相遅れに対処する
    t_first = np.maximum(est_rpeaks[0],ref_rpeaks[0])
    est_rpeaks = est_rpeaks[(t_first<=est_rpeaks)]
    ref_rpeaks = ref_rpeaks[(t_first<=ref_rpeaks)]
    est_rpeaks = est_rpeaks - est_rpeaks[0]
    ref_rpeaks = ref_rpeaks - ref_rpeaks[0]

    # RRIを取得
    est_rri = est_rpeaks[1:]-est_rpeaks[:-1]
    est_rpeaks = est_rpeaks[1:]
    ref_rri = ref_rpeaks[1:]-ref_rpeaks[:-1]
    ref_rpeaks = ref_rpeaks[1:]
    input_peaknum = ref_rri.size
    print("Input  REF RRI:{}, EST RRI:{}".format(ref_rri.size, est_rri.size))

    if est_rpeaks.size != ref_rpeaks.size:
        # Estimate peaks内で閾値より大きく外れたデータを削除
        median_rri = signal.medfilt(est_rri, 5)# median filter
        detrend_est_rri = est_rri - median_rri
        index_outlier = np.where(np.abs(detrend_est_rri) > (threshold*1000))[0]
        print("{} point detected".format(index_outlier.size))
        if index_outlier.size > 0:
            flag = np.ones(len(est_rri), dtype=bool)
            flag[index_outlier.tolist()] = False
            est_rpeaks = est_rpeaks[flag]
            est_rri = est_rri[flag]
            
            # リファレンスと比較して，大きく外れたデータを検出
            ref_index = []
            for i,i_rpeak in enumerate(ref_rpeaks):
                # リスト要素と対象値の差分を計算し最小値のインデックスを取得
                idx = np.abs(est_rpeaks-i_rpeak).argmin()
                if np.abs(est_rpeaks[idx]-i_rpeak) <= (threshold*1000):
                    ref_index.append(i)
            ref_rpeaks = ref_rpeaks[ref_index]
            ref_rri = ref_rri[ref_index]
            
    print("Output REF RRI:{}, EST RRI:{}".format(ref_rri.size, est_rri.size))
    
    error_rate = (input_peaknum-ref_rri.size)/input_peaknum
    print("Error Rate: {}%".format(100*error_rate))
    return ref_rri,est_rri,error_rate

if __name__ == "__main__":
    path = r"D:\rPPGDataset\Analysis\luminance\shizuya\2021-01-05 18-45-41.248194 Front And Celling 700lux rPPG Signals.csv"
    rppg_signal = np.loadtxt(path,delimiter=",")[:,0]
    fs = 30
    
