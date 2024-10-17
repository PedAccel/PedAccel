#This Script generates a Principle Component Analysis Plot
#give PCA access to modules folder
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from scipy.io import loadmat
import tsfresh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Modules import Actigraph_Metrics, Smoothing_Functions

# Load Data
data_dir = './PatientData/9'
filename = 'pt9_win5_5.mat'

data_path = Path(data_dir)
fp = data_path/filename
data = loadmat(fp)

x_mag = data["x_mag"]
SBS = data["sbs"]

#Create feature arrays. To Do: Replace this with automatically generated 100 feature array. 
sbs = []
for i in SBS[0]:
    sbs.append(i)
abs_energy = []
abs_max = []
count_above_mean = []
count_below_mean = []
std = []
mean = []
number_peaks = []
sample_entropy = []
sum_of_changes = []
kurtosis = []
complexity = []
mean_abs_change = []
variance = []
auc = []

print('Next is feature extraction')
#Fill in feature arrays

for signal in x_mag:

    #MADtime is x axis for area under curve function
    MADx = []

    #Get MAD and ENMO data
    MAD = Actigraph_Metrics.VecMag_MAD(signal)
    ENMO = Smoothing_Functions.ENMO(signal)

    #Fill in x axis array for MAD
    for j in range(len(MAD)):
        MADx.append(j/100)

    #choose the type of signal to extract features on
    #i can be signal, MAD, or ENMO
    i = MAD

    #Calculate Features
    abs_energy.append(tsfresh.feature_extraction.feature_calculators.abs_energy(i))
    abs_max.append(tsfresh.feature_extraction.feature_calculators.absolute_maximum(i))
    count_above_mean.append(tsfresh.feature_extraction.feature_calculators.count_above_mean(i))
    count_below_mean.append(tsfresh.feature_extraction.feature_calculators.count_below_mean(i))
    std.append(np.std(i))
    mean.append(np.mean(i))
    number_peaks.append(tsfresh.feature_extraction.feature_calculators.number_peaks(i, 1000))
    sum_of_changes.append(tsfresh.feature_extraction.feature_calculators.absolute_sum_of_changes(i))
    kurtosis.append(tsfresh.feature_extraction.feature_calculators.kurtosis(i))
    complexity.append(tsfresh.feature_extraction.feature_calculators.lempel_ziv_complexity(i,60))
    mean_abs_change.append(tsfresh.feature_extraction.feature_calculators.mean_abs_change(i))
    variance.append(tsfresh.feature_extraction.feature_calculators.variance(i))
    auc.append(Actigraph_Metrics.calc_area_under_curve(MADx, MAD))

print('Making data frame')

data = {'abs_energy' : abs_energy,'abs_max' : abs_max,'count_above_mean' : count_above_mean,'count_below_mean' : count_below_mean,
       'std' : std,'mean' : mean,'number_peaks' : number_peaks, 'Sum of changes': sum_of_changes, 'Kurtosis': kurtosis, 
       'Complexity': complexity, 'mean abs changhe': mean_abs_change, 'Variance': variance, 'auc': auc, 'SBS' : sbs}

df = pd.DataFrame(data)


print('df made')

#Normalize the data, remove SBS as a feature for the analysis
x = df[df.columns[:-1]].values
print("df with SBS removed for analysis:")
print(df[df.columns[:-1]].head(5))

x = StandardScaler().fit_transform(x) # normalizing the features

#mean should be zero, std should be 1
print(np.mean(x),np.std(x))


#Store normalized data
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_df = pd.DataFrame(x,columns=feat_cols)
print(normalised_df.head(5))
print('PCA calculations')

#PCA
pca_actigraphy = PCA(n_components=4)
principalComponents_actigraphy = pca_actigraphy.fit_transform(x)
principal_actigraphy_Df = pd.DataFrame(data = principalComponents_actigraphy
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4'])

#print(principal_actigraphy_Df.head(20))
print('Explained variation per principal component: {}'.format(pca_actigraphy.explained_variance_ratio_))
print('Plotting PCA')


#Plot PCA
plot = plt.figure(figsize=(12,12))
plt.figure(figsize=(12,12))
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Actigraphy and SBS",fontsize=20)


#Color code points based on SBS
for i in range(len(df['SBS'])):
    if df['SBS'][i] == -1:
        color = 'purple'
    if df['SBS'][i] == 0:
        color = 'blue'
    if df['SBS'][i] == 1:
        color = 'orange'
    if df['SBS'][i] == 2: 
        color = 'red'
    plt.scatter(principal_actigraphy_Df.loc[i, 'principal component 1'], principal_actigraphy_Df.loc[i, 'principal component 2'], c = color, s = 50)
    
#Manually create a legend
neg1 = mlines.Line2D([], [], color='purple', marker='o', ls='', label='SBS -1')
zero = mlines.Line2D([], [], color='blue', marker='o', ls='', label='SBS 0')
one = mlines.Line2D([], [], color='orange', marker='o', ls='', label='SBS 1')
two = mlines.Line2D([], [], color='red', marker='o', ls='', label='SBS 2')
plt.legend(handles=[neg1, zero, one, two])
plt.show()

