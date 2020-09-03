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

dirpath = ""
tspath = ""

files = []
i = 0
for filename in os.listdir(dirpath):
    if os.path.isfile(os.path.join(dirpath, filename)): #ファイルのみ取得
        files.append(filename)

# ファイル名からフレーム時刻を取得
timestamps = []
for file in files:
    file_param = re.split('[_ ]', file)
    timestamp = re.split('[-]', file_param[1])
    # Str to int
    timestamp = [float(s) for s in timestamp]
    timestamps.append(timestamp[0]*60**2 + timestamp[1]*60 + timestamp[2])

data_time = np.array(timestamps)
data_timediff = 1/np.diff(data_time)
print(np.mean(data_timediff))
np.savetxt(tspath,data_time)