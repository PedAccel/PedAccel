#%%
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tsfel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.lines as mlines
from scipy.io import loadmat

from Modules import Actigraph_Metrics

#%%
# Load Data
data_dir = './PatientData/9'
filename = 'Patient9_5MIN_DSW_AllSBS.mat'
fp = os.path.join(data_dir, filename)

data = loadmat(fp)
x_mag = data["x_mag"]
SBS = data["sbs"]

# Generate configuration file for feature extraction
cfg_file = tsfel.get_features_by_domain()

#%%
# Extract features and restructure data
features_list = []
sbs_list = []
for i in range(x_mag.shape[0]):
    signal = Actigraph_Metrics.VecMag_MAD(x_mag[i,:],100)
    features = tsfel.time_series_features_extractor(cfg_file, signal, fs=100, verbose=0)
    features_list.append(features)
    sbs_list.append(SBS[0][i])

#%%
# Convert features and SBS scores to DataFrame
features_array = np.array(features_list).reshape(-1, 389)
df_features = pd.DataFrame(features_array)
df_features.columns = ['feature_' + str(col) for col in df_features.columns]

df_sbs = pd.DataFrame({'SBS': sbs_list})

# Concatenate features and SBS scores
df = pd.concat([df_sbs, df_features], axis=1)
x = df_features.values
y = df['SBS'].values

#%%
# Normalize features
x = df.iloc[:, 1:].values
x_normalized = StandardScaler().fit_transform(x)
df_normalized = pd.DataFrame(x_normalized, columns=df_features.columns)

# Calculate variance of each normalized feature
normalized_feature_variances = df_normalized.var()

# Select top 10 features with highest variance
x = df_normalized[normalized_feature_variances.nlargest(10).index].values

#%%
# Perform PCA
pca_actigraphy = PCA(n_components=10)
principalComponents_actigraphy = pca_actigraphy.fit_transform(x)
principal_actigraphy_Df = pd.DataFrame(data=principalComponents_actigraphy,
                                      columns=['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4',
                                               'principal component 5', 'principal component 6', 'principal component 7',
                                               'principal component 8', 'principal component 9', 'principal component 10'])

print('Explained variation per principal component: {}'.format(pca_actigraphy.explained_variance_ratio_))

# Plot PCA for each principal component
for component in range(pca_actigraphy.n_components_):
    plt.figure(figsize=(8, 6))
    plt.xlabel(f'Principal Component - {component+1}', fontsize=12)
    plt.title(f'Principal Component Analysis of Actigraphy and SBS - {component + 1}', fontsize=14)
    
    for i in range(len(df['SBS'])):
        if df['SBS'][i] == -1:
            color = 'purple'
            y_offset = -1 
        elif df['SBS'][i] == 0:
            color = 'blue'
            y_offset = 0
        elif df['SBS'][i] == 1:
            color = 'orange'
            y_offset = 1
        elif df['SBS'][i] == 2:
            color = 'red'
            y_offset = 2
        plt.scatter(principal_actigraphy_Df.loc[i, f'principal component {component+1}'], y_offset, 
                    c=color, s=50)
        
    
    # Manually create a legend
    neg1 = mlines.Line2D([], [], color='purple', marker='o', ls='', label='SBS -1')
    zero = mlines.Line2D([], [], color='blue', marker='o', ls='', label='SBS 0')
    one = mlines.Line2D([], [], color='orange', marker='o', ls='', label='SBS 1')
    two = mlines.Line2D([], [], color='red', marker='o', ls='', label='SBS 2')
    plt.legend(handles=[neg1, zero, one, two])
    
    plt.show()
# %%
