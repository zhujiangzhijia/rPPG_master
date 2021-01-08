"""
ImprovedSoftSig
Dentsu university 
"""
# coding: utf-8
# POSMethod(rgb_components, WinSec=1.6, LPF=0.7, HPF=2.5, fs=30, filter = False)

# Simpler Proposed Method (no physiological conditions)
def ImprovedSoftSig(R,G,B,fs):
    # R,G,B ... evenly spaced time intervals
    # RGB信号は正規化されている。
    # RGB信号は10sなど時間で区切られている.
    deltaf  = 0.4
    L       = ceil(length(G)/fs) # [s]計測長
    padL    = 10*ceil(L/10) 
    PRVband = round(deltaf*padL)
    k       = 0
    step    = 0.1
    ksize   = (2/step+1)^2*(1/step+1)
    SN      = zeros(1,ksize)
    v       = zeros(3,ksize)


    for v2 in range(0,1,step): # v2: G
        for v1 in range(-1,1,step): # v1:R
            for v3 in range(-1,1,step): # v3: B
                if ( abs(v2) >= abs(v1) ) and ( abs(v2) >= abs(v3) ):
                    weightedsum = v1 * R + v2 * G + v3 * B
                    
                    spec = mypsd(weightedsum,fs,padL) # xpsd
                    [~,I] = max(spec(0.5*padL+1 : 2*padL+1,2))
                    I = 0.5*padL+I
                    k = k + 1
                    # SN(1,k) = M / (sum(spec(:,2))-M) %<-SoftSig
                    SN(1,k) = trapz(1/padL,spec(I-PRVband:I+PRVband,2)) / (trapz(1/padL,spec(1:I-PRVband,2))+trapz(spec(I+PRVband:end,2)));
                    # ?? SN(1,k) = trapz(1/padL,spec(I-PRVband:I+PRVband,2)) / (trapz(1/padL,spec(1:I-PRVband,2))+trapz(spec(I+PRVband:end,2)));
                    v(:,k) = [v1,v2,v3]

    [~,maxSN] = max(SN)
    vR = v(1,maxSN);
    vG = v(2,maxSN);
    vB = v(3,maxSN);
    return [vR,vG,vB]


# PSD (FFT)
function pxx = mypsd(x,fs,padlength)
x = padarray(x,padlength*fs-length(x),'post');

xdft = fft(x);
xdft = xdft(1:length(x)/2+1);
xpsd = (1/(fs*length(x))) * abs(xdft).^2;
xpsd(2:end-1) = 2*xpsd(2:end-1);
freq = (0:fs/length(x):fs/2).';