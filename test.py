import numpy as np
import scipy.fftpack as spfft
import matplotlib.pyplot as plt
f0 = 2000
fs = 3000
N = 1000
addnum = 5.0

def create_sin_wave(amplitude,f0,fs,sample):
    wave_table = []
    for n in np.arange(sample):
        sine = amplitude * np.sin(2.0 * np.pi * f0 * n / fs)
        wave_table.append(sine)
    return wave_table

wave1 = create_sin_wave(1.0,f0,fs,N)

X = spfft.fft(wave1[0:N])
freqList = spfft.fftfreq(N, d=1.0/ fs)
amplitude = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]  # 振幅スペクトル


# 振幅スペクトルを描画
plt.plot(freqList, amplitude, marker='.', linestyle='-',label = "fft plot")
plt.xlabel("frequency [Hz]")
plt.ylabel("amplitude")
plt.show()

