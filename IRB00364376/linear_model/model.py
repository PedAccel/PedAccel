# %%
from scipy.io import loadmat, savemat
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import Lasso, Ridge, LinearRegession
from sklearn.neural_network import MLPRegressor
#%%
fold = 1
# %%
X_train = (loadmat(f'fold{fold}.mat')['X_train'])
y_train = (loadmat(f'fold{fold}.mat')['y_train']) 
X_test = (loadmat(f'fold{fold}.mat')['X_test'])  
y_test = (loadmat(f'fold{fold}.mat')['y_test'])

# %%
# training
# model = MLPRegressor(hidden_layer_sizes=(50, ))
model = Ridge(alpha=1e-2)
model.fit(X_train, np.squeeze(y_train))
y_predict = model.predict(X_test)

# %%
plt.scatter(np.squeeze(y_test), y_predict[:])
plt.axis([-4, 3, -4, 3])
plt.xlabel('actual')
plt.ylabel('predict')
# %%
print(metrics.r2_score(np.squeeze(y_test), y_predict))
# %%
