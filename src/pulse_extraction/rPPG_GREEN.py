"""
GREEN_VERKRUYSSE The Green-Channel Method from: 
Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445. DOI: 10.1364/OE.16.021434
"""
# coding: utf-8
import numpy as np
import pandas as pd
from .. import preprocessing


def GreenMethod(rgb_signals, LPF=0.7, HPF=2.5, fs=30):
    # Green Channel
    rppg = rgb_signals[:, 1]
    # Filter, Normalize
    filtered_rppg = preprocessing.ButterFilter(rppg, LPF, HPF, fs)
    # Moving Average
    smooth_rppg = preprocessing.MovingAve(filtered_rppg, num=30)
    return smooth_rppg

if __name__ == "__main__":
    print('__package__: {}, __name__: {}'.format(
    __package__, __name__))
    import matplotlib.pyplot as plt
    bgr_component = pd.read_csv(r"C:\Users\akito\Desktop\testets.csv",usecols=[3,4,5],header=0,index_col=0)
    
    plt.plot(GreenMethod(bgr_component))

    plt.show()
