# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 20:33:57 2020

@author: yuya4

顔の領域を平均化する

"""

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import statsmodels.api as sm
from scipy import interpolate
from scipy.stats import zscore



rcParams["figure.figsize"] = 18, 8
csv_path = 'data/1_ROI_old.csv'
camera_num = 1
df = pd.read_csv(csv_path)
camera_total_num = len([column for column in df.columns if 'camera' in column])

# データ出力用の箱を作成
df_out = pd.DataFrame(columns = ['frame'])


for camera_num in range(1, camera_total_num+1):
    print('--------')
    print(f'camera{camera_num}')
    print('--------')

    data = df[f'camera{camera_num}']
    data = data.dropna()
    res = sm.tsa.seasonal_decompose(data.values, freq=15)

    plt.figure(1)
    res.plot()

    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid

    detrend = seasonal + residual

    detrend_series = pd.Series(detrend)
    # NaNを除去
    detrend_number = detrend_series.dropna()

    # 正規化
    data_std = zscore(detrend_number)

    N = len(detrend_number)             # サンプル数
    dt = 1/30         # サンプリング間隔

    # 軸の計算
    t = np.arange(0, N*dt, dt)  # 時間軸
    freq = np.linspace(0, 1.0/dt, N)  # 周波数軸

    fc = 3  # カットオフ周波数
    fc_under = 0.5
    fs = 1 / dt  # サンプリング周波数
    fm = (1/2) * fs  # アンチエリアジング周波数
    fc_upper = fs - fc  # 上側のカットオフ　fc～fc_upperの部分をカット

    f = data_std
    # f = detrend_2
    # 元波形をfft
    F = np.fft.fft(f)

    # 正規化 + 交流成分2倍
    # F = F/(N/2)
    # F[0] = F[0]/2

    # アンチエリアジング
    # F[(freq > fm)] = 0 + 0j

    # 元波形をコピーする
    G = F.copy()

    # ローパス
    G[((freq > fc) & (freq < fc_upper))] = 0 + 0j

    # ハイパス
    G[(0 < freq) & (freq < fc_under)] = 0 + 0j

    # 高速逆フーリエ変換
    g = np.fft.ifft(G)

    # 実部の値のみ取り出し
    g = g.real

    # プロット確認
    # plt.subplot(221)
    # plt.plot(t, f)
    #
    # plt.subplot(222)
    # plt.plot(freq, F)
    # plt.xlim([0, 15])
    #
    # plt.subplot(223)
    # plt.plot(t, g)
    #
    # plt.subplot(224)
    # plt.plot(freq, G)
    # plt.xlim([0, 15])

    ############################
    # Interpolate & UpSampling #
    ############################

    resamp_f = 1000
    fps = 30
    resamp_num = int(g.shape[0] * resamp_f / fps)

    np_data = np.array(g)
    x = np.arange(len(np_data))
    tck = interpolate.interp1d(x, np_data)
    resamp = np.linspace(0, len(np_data)-1, resamp_num)
    resamp_pulse = tck(resamp)

    plt.figure(1)
    plt.plot(x, np_data)
    plt.show()

    plt.figure(2)
    plt.plot(resamp, resamp_pulse)
    plt.show()

    framenumber = np.arange(len(resamp_pulse))

    df_out = pd.concat([df_out, pd.Series(resamp_pulse, name=f'camera{camera_num}')], axis=1)
    if len(framenumber) >= len(df_out['frame']):
        df_out['frame'] = pd.Series(framenumber)
        df_out['frame'].astype(np.int16)

# データの保存
df_out.to_csv(f'2_takase_0123_frame_fft_0.csv.csv',index=False)
