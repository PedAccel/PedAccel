#%%
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy.io import loadmat
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import tsfel
from operator import itemgetter

from Modules import Actigraph_Metrics

# Load Data
data_dir = './PatientData/9'
filename = 'pt9_5min_twoside.mat'

data_path = Path(data_dir)
fp = data_path/filename
data = loadmat(fp)

x_mag = data["x_mag"]
SBS = data["sbs"]


#%%
# Generate configuration file for feature extraction
cfg_file = tsfel.get_features_by_domain()
print(x_mag.shape)

#%%
# Extract features and restructure data
features_list = []
sbs_list = []
for i in range(x_mag.shape[0]):
    signal = Actigraph_Metrics.VecMag_MAD(x_mag[i,:],100)
    #signal = np.array(x_mag[i,:])-1
    #signal = x_mag[i,:]
    features = tsfel.time_series_features_extractor(cfg_file, signal, fs=100, verbose=0)
    features_list.append(features)
    sbs_list.append(SBS[0][i])

#%%
# Convert features and SBS scores to DataFrame
#Find number of features
#print((features_list))
features_array = np.array(features_list).reshape(-1, 389)
df_features = pd.DataFrame(features_array)
#list comprehension for column names
columns = [col for col in list(features_list[0])]
df_features.columns = columns

#Pearson Correlation Coefficient
CCoeff = []
for i in columns:
    y = sbs_list
    x = list(df_features[i])
    corr, _ = pearsonr(y, x)
    CCoeff.append(np.abs(corr))
my_dict = dict(zip(list(columns), list(CCoeff)))

# functional
clean_dict = filter(lambda k: not math.isnan(my_dict[k]), my_dict)
# dict comprehension
clean_dict = {k: my_dict[k] for k in my_dict if not math.isnan(my_dict[k])}

#Retrieve N features with best correlation coefficient  
# Initialize N
N = 10
 
# N largest values in dictionary
# Using sorted() + itemgetter() + items()
res = dict(sorted(clean_dict.items(), key=itemgetter(1), reverse=True)[:N])

# printing result
print("The top N value pairs are " + str(res))
#_________________________________________________________________________________
#Plot a histogram

y = list(res.keys())
x = list(res.values()) #price
 
# Figure Size
fig, ax = plt.subplots(figsize =(16, 9))
 
# Horizontal Bar Plot
ax.barh(y, x)
 
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
 
# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
ax.grid(color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
 
# Show top values 
ax.invert_yaxis()

#set x axis range
ax.set_xlim([.8*min(x),1.1*max(x)])
 
# Add Plot Title
ax.set_title('Correlation between top features and SBS',
             loc ='left', )
 
# Show Plot
plt.show()
#__________________________________________________________________________________
#Make PCA using these features
"""
df_new_features = df_features[res.keys()]

df_sbs = pd.DataFrame({'SBS': sbs_list})
# Concatenate features and SBS scores
df = pd.concat([df_sbs, df_new_features], axis=1)
print("better data\n")
print(df_new_features.head(10))

# Normalize the data
x = df_new_features[df_new_features.columns].values
print("df with SBS removed for analysis:")

x = StandardScaler().fit_transform(x) # normalizing the features

#mean should be zero, std should be 1
print(np.mean(x), np.std(x))


#Store normalized data
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_df = pd.DataFrame(x,columns=feat_cols)
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
        continue
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
"""