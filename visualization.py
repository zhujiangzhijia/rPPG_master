# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv("outputtest.csv",usecols=range(4),header=0,index_col=0)
data.plot()
plt.show()