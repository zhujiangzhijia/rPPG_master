# -*- coding: utf-8 -*-
"""

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fftpack import fft
from scipy.stats import zscore


def signal_fft(sign, N):  # FFTするsignal長と窓長Nは同じサンプル数に固定する
    """Fft siganl."""
    spectrum = fft(sign)  # フーリエ変換
    spectrum_abs = np.abs(spectrum)  # 振幅を元に信号に揃える
    half_spectrum = spectrum_abs[:int(N/2)]
    half_spectrum[0] = half_spectrum[0] / 2  # 直流成分（今回は扱わないけど）は2倍不要

    return half_spectrum

def calc_RRI(peak_frame):
    # ピークをフレームから時間に直す
    R_x = (peak_frame/fs).reshape(len(peak_frame/fs))
    RRI = np.array([R_x[idx+1] - R_x[idx] for idx in range(len(R_x)-1)])
    ave_RRI = np.mean(RRI)

    return R_x, RRI, ave_RRI


fs = 1000

df = pd.read_csv('data/1_teraki_Rwave.csv')
df_out = pd.DataFrame()
camera = []
for camera_num in range(1, df.shape[1]): # camera1のカラムを抽出
    camera.append(df.iloc[:, camera_num])

# 配列に直す
camera_array = np.array(camera)

fig, ax = plt.subplots(len(camera_array), 2, figsize=(10,5*len(camera_array)))
for i in range(len(camera_array)):
    signal_array = zscore(camera_array[i])
    fft_spectrum = signal_fft(signal_array, len(signal_array))

    f = np.arange(0, fs/2, (fs/2)/fft_spectrum.shape[0])  # 横軸周波数軸[Hz]

    low_th = 0.8
    high_th = 1.65
    # plt.figure(0)
    # plt.semilogx(f, fft_spectrum)
    # plt.xlim(low_th, high_th)

    fc = f[np.where(np.round(f, 1)==low_th)[0][0]+np.argmax(fft_spectrum[np.where(np.round(f, 1)==low_th)[0][0]:np.where(np.round(f, 2)==high_th)[0][0]])]

    peak_list = []
    peak_range_start = []
    peak_range_end = []
    idx = 0
    range_rate = 0.4
    peak_frame = 0

    while peak_frame <= len(signal_array):
        if idx == 0:
            target = signal_array[0*fs:1*fs]
            peak_frame = np.argmax(target)
            peak_list.append(peak_frame)
            peak_range_end.append(1*fs)

        else:
            target = signal_array[int(peak_list[idx-1]+(1-range_rate)/fc*fs):int(peak_list[idx-1]+(1+range_rate)/fc*fs)]
            tmp_peak_list=[]
            dif_list=[]
            count_peak = 0

            for frame in range(len(target)-2):
                plus_flag = True if target[frame+1]-target[frame] >=0 else False
                if plus_flag:
                    minus_flag = True if target[frame+2]-target[frame+1] <0 else False
                if plus_flag and minus_flag:
                    peak_frame = frame+int(peak_list[idx-1]+(1-range_rate)/fc*fs)
                    tmp_peak_list.append(peak_frame)
                    count_peak += 1

            if count_peak > 1:
                for peak in tmp_peak_list:
                    dif_list.append(np.abs(peak_list[idx-1]+1/fc*fs - peak))
                peak_frame = tmp_peak_list[np.where(dif_list ==np.min(dif_list))[0][0]]

            elif count_peak == 0:
                peak_frame = int(peak_list[idx-1]+(1-range_rate)/fc*fs) + (int((1+range_rate)/fc*fs)- int((1-range_rate)/fc*fs))//2
                # print('interpolate', peak_frame)

            if peak_frame<=len(signal_array):
                # print(peak_frame)
                peak_list.append(peak_frame)
            if int(peak_list[idx-1]+(1-range_rate)/fc*fs)<=len(signal_array):
                peak_range_start.append(int(peak_list[idx-1]+(1-range_rate)/fc*fs))
            if int(peak_list[idx-1]+(1+range_rate)/fc*fs)<=len(signal_array):
                peak_range_end.append(int(peak_list[idx-1]+(1+range_rate)/fc*fs))
        # plt.figure(idx+1)
        # plt.plot(target)
        idx += 1

    R_x, RRI, ave_RRI = calc_RRI(np.array(peak_list))
    print(f'camera{i}, average_RRI:', ave_RRI)

    t = np.linspace(0, len(signal_array)/fs, len(signal_array))
    x_start = 10
    x_end = 20

    ax[i,0].plot(t, signal_array, label='signal')
    ax[i,0].plot(t[peak_list], signal_array[peak_list], 'ro', label='peak')
    ax[i,0].vlines(t[peak_range_start], -5, 5, "red", linestyles='dashed')
    ax[i,0].vlines(t[peak_range_end], -5, 5, "blue", linestyles='dashed')
    ax[i,0].set_xlabel('Ri[s]')
    ax[i,0].set_ylabel('RRI[s]')
    ax[i,0].set_xlim(x_start, x_end)
    ax[i,0].set_ylim(np.min(signal_array[x_start*fs:x_end*fs])-1, np.max(signal_array[x_start*fs:x_end*fs])+1)

    ax[i,1].plot(R_x[1:], RRI)
    ax[i,1].set_xlabel('Ri[s]')
    ax[i,1].set_ylabel('RRI[s]')
    ax[i,1].set_ylim(0.4, 1.6)

    df_out[f'R_x_{i+1}'] = pd.Series(R_x[1:])
    df_out[f'RRI_{i+1}'] = pd.Series(RRI)

df_out.to_csv('result/test.csv', index=False)
