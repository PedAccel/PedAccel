#%%
#give PCA access to modules folder
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.insert(1, r"C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\Modules")
sys.path.insert(0, r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\Modules')

print(sys.version) 
import sysconfig; 
#Print where python looks for packages and where packages are downloaded. pip -V is where pip is installed to. 
#pip show 'package name'
print(sysconfig.get_paths()["purelib"])
#use pip install --target=C:\Users\jakes\.virtualenvs\DT6Analysis\Lib\site-packages package name to install to specific folder
from operator import itemgetter
from math import isnan
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.lines as mlines
from scipy.io import loadmat
import os
import Actigraph_Metrics
import tsfel
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import seaborn as sns

# Load Data
# os.chdir(r"C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData\Patient9")
os.chdir(r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData\Patient9')
#%%
filename = 'Patient9_5MIN_SW_AllSBS.mat'

x_mag = (loadmat(filename)["x_mag"])
SBS = loadmat(filename)["sbs"]

#signal = Actigraph_Metrics.VecMag_MAD(x_mag[i,:],100)
#signal = np.array(x_mag[i,:])-1
#signal = x_mag[i,:]

SBS_Changes = []
AUC_Changes = []
PEAK_PEAK_Changes = []
Mean_Changes = []

AUC = []
PEAK_PEAK = []
Mean = []

# Populate array with changes in SBS
for i in range(len(SBS[0])-1):
    state = SBS[0][i]-SBS[0][i+1]
    if(state > 0):
        SBS_Changes.append(-1)
    else: 
        SBS_Changes.append(1)
print(SBS_Changes)

#Generate Peak to peak, mean, auc data
for i in range(x_mag.shape[0]):
    signal = Actigraph_Metrics.VecMag_MAD(x_mag[i,:],100)
    #signal = np.array(x_mag[i,:])-1
    #signal = x_mag[i,:]
    x = []
    for i in range(len(signal)):
        x.append(i)
    Mean.append(np.mean(signal))
    AUC.append(Actigraph_Metrics.calc_area_under_curve(x,signal))
    PEAK_PEAK.append(max(signal)-min(signal))
    
# define threshold
threshold = .0025
# Populate array with changes in SBS
count = 0
for i in range(len(SBS[0])-1):
    state1 = Mean[i] - Mean[i+1]
    if(state1 > threshold):
        Mean_Changes.append(-1)
    else: 
        Mean_Changes.append(1)
    if(Mean_Changes[i] == SBS_Changes[i]):
        count+=1
print(f"Percentage of same changes with Mean is: {100*(count/len(SBS_Changes))}%")

# Populate array with changes in SBS
count = 0
for i in range(len(SBS[0])-1):
    state1 = PEAK_PEAK[i] - PEAK_PEAK[i+1]
    if(state1 > threshold):
        PEAK_PEAK_Changes.append(-1)
    else: 
        PEAK_PEAK_Changes.append(1)
    if(PEAK_PEAK_Changes[i] == SBS_Changes[i]):
        count+=1
print(f"Percentage of same changes with Peak to Peak is: {100*(count/len(SBS_Changes))}%")

# Populate array with changes in SBS
count = 0
for i in range(len(SBS[0])-1):
    state1 = AUC[i] - AUC[i+1]
    if(state1 > threshold):
        AUC_Changes.append(-1)
    else: 
        AUC_Changes.append(1)
    if(AUC_Changes[i] == SBS_Changes[i]):
        count+=1
print(f"Percentage of same changes with AUC is: {100*(count/len(SBS_Changes))}%")

# %%
