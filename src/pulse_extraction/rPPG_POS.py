"""
POS_WANG The Plane Orthogonal to Skin-Tone (POS) Method from:
　Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG.


"""
# coding: utf-8
import numpy as np

from .. import preprocessing


def POSMethod(rgb_components, WinSec=1.60, LPF=0.7, HPF=2.5, fs=30):
    """
    CHROM method
    WinSec :was a 32 frame window with 20 fps camera
    """
