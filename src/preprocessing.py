"""
RGB値の前処理を担当する
前処理には，BPFが含まれる
"""
# coding: utf-8
import numpy as np
from scipy import signal

def ButterFilter(data, lowcut, highcut, fs, order=3):
    """
    Butter Band pass filter
    doc:
     https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
     https://www.it-swarm.dev/ja/python/scipysignalbutter%E3%81%A7%E3%83%90%E3%83%B3%E3%83%89%E3%83%91%E3%82%B9%E3%83%90%E3%82%BF%E3%83%BC%E3%83%AF%E3%83%BC%E3%82%B9%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF%E3%83%BC%E3%82%92%E5%AE%9F%E8%A3%85%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95/1067792786/
    scipy:0.4.1verと引数が変わっているようなので注意
    ・最小の方は少し振幅が小さくなる
    ・SOSの方が安定しているそう
    """
    detrend = data - np.mean(data)
    sos = ButterBandpass(lowcut, highcut, fs, order=order)
    y = signal.sosfilt(sos, data)
    return y

def ButterBandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def FGTransform(rgb_components):
    """
    Transform from: 
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. 
    """
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        sos = ButterBandpass(lowcut, highcut, fs, order=order)
        w, h = signal.sosfreqz(sos, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')
    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = ButterFilter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()