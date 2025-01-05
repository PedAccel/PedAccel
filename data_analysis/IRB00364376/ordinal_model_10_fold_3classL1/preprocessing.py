#%%
from scipy.io import loadmat, savemat
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import torch.optim as optim
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import tsfel
from sklearn.model_selection import GroupKFold
from sklearn import metrics
import os
import matplotlib.pyplot as plt
#%%
def ordinal_labels(y, num_classes = None):
    if not num_classes:
        num_classes = np.max(y) + 1
    range_values = np.arange(num_classes - 1)[None, :]
    range_values = np.tile(range_values, [y.shape[0], 1])
    ordinal_label = np.where(range_values < y, 1, 0)
    return ordinal_label
# %%
X = loadmat('../DONOTPUSH/data.mat')['X']
#%%
y = loadmat('../DONOTPUSH/data.mat')['y']
y[y < -2] = -1
y[y == -1] = 0
y[y > 1] = 1
y= y+1
ids = loadmat('../DONOTPUSH/data.mat')['group']
#%%
selected = np.squeeze(y != y.round)
X = X[selected, :]
ids = np.squeeze(ids)[selected]
y = np.squeeze(y)[selected]
#%%
X_c = X
y_c = y
# %%
# tsfel (https://tsfel.readthedocs.io/en/latest/)
cfg_file = tsfel.get_features_by_domain()
del cfg_file['spectral']["LPCC"]
del cfg_file["spectral"]["MFCC"]
del cfg_file["spectral"]["Spectral roll-on"]
fs = 1/60

# %%
x_features = np.concatenate([np.vstack([tsfel.time_series_features_extractor(cfg_file, X[i, k, :], fs = fs, verbose = 0).to_numpy() for i in range(X.shape[0])])[:, :, None] for k in range(X.shape[1]-2)], axis = 2)
features = tsfel.time_series_features_extractor(cfg_file, X[0, 0, :], fs = fs, verbose = 0)
features = features.columns.to_list()
# %%
x_features = x_features.reshape(x_features.shape[0], -1)
#%%
X = np.concatenate((x_features, X[:, 3, :].mean(axis = 1)[:, None], X[:, 4, :].mean(axis = 1)[:, None]), axis=1)
y = ordinal_labels(np.squeeze(y).reshape(-1, 1),3)
# %%
labels = ['HR', 'RR', 'SPO2']
all_features = []
for l in labels:
    waveform_features = [l + '_' + feat for feat in features]
    all_features += waveform_features
#%%
all_features.append('age')
all_features.append('vent')
#%%
X = X.astype(float)
nan_feature = np.any(np.isnan(X), axis=0)
#%%
n_nan_features = [all_features[i] for i in range(len(all_features)) if not i in np.where(nan_feature)[0].tolist()]
#%%
X = X[:, ~np.any(np.isnan(X), axis=0)]
#%%
for repeat in range(10):
    folder = f"cv{repeat}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    kf = GroupKFold(n_splits=10)
    for i, (train_idx, test_idx) in enumerate(kf.split(X,y, groups=ids)):
        x_train_data = X[train_idx, :]
        y_train_data = y[train_idx, :]
        x_test_data = X[test_idx, :]
        y_test_data = y[test_idx, :]
        # normalizing input features
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train_data)
        x_test = scaler.transform(x_test_data)
        x_wave = X_c[test_idx, :, :]
        test_ids = ids[test_idx]
        filename = folder + (f"fold{i}.mat")
        savemat(filename, {'X_train':x_train, 'y_train':y_train_data, 'X_test':x_test, 'y_test':y_test_data, 'X_test_wave':x_wave, 'test_ids':test_ids})
# %%
import json

with open('feat.json', 'w') as f:
    json.dump(n_nan_features, f)
# %%

#%%
folder = "distributions/"
for start in range(0, X.shape[1], 9):
    end =  start + 9 if start + 9 < X.shape[1] else X.shape[1]
    fig, axis = plt.subplots(3, 3, figsize=(12, 12))
    axis = axis.ravel()
    for idx, ax in enumerate(axis, start= start):
        ax.hist(X[:, idx])
        ax.set_title(all_features[idx])
    fig.savefig(folder + f"img{start}.png")
    plt.close(fig)

#%%
for repeat in range(10):
    folder = f"cv{repeat}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    kf = GroupKFold(n_splits=10)
    for i, (train_idx, test_idx) in enumerate(kf.split(X,y, groups=ids)):
        x_train_data = X[train_idx, :]
        y_train_data = y[train_idx, :]
        x_test_data = X[test_idx, :]
        y_test_data = y[test_idx, :]
        # normalizing input features
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train_data)
        x_test = scaler.transform(x_test_data)

        filename = folder + (f"fold{i}.mat")
        savemat(filename, {'X_train':x_train, 'y_train':y_train_data, 'X_test':x_test, 'y_test':y_test_data})
# %%
import json

with open('feat.json', 'w') as f:
    json.dump(n_nan_features, f)
# %%
