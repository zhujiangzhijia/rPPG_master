"""
Opensignalで計測したデータを
処理する
"""
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def plot_signal(path):
    data = np.loadtxt(path)
    ECGsignal = data[:, -2]
    BTNsignal = data[:, -1]
    stindex = np.where(BTNsignal == 0)[0][0]
    plt.plot(ECGsignal[stindex:])
    plt.show()
    
def ECG(path):
    data = np.loadtxt(path)
    ECGsignal = data[:, -2]
    BTNsignal = data[:, -1]
    stindex = np.where(BTNsignal == 0)[0][0]
    return ECGsignal[stindex:]

if __name__ == "__main__":
    path = "./video/motionrotation_201808080163_2020-05-06_17-07-14.txt"
    plot_signal(path)
