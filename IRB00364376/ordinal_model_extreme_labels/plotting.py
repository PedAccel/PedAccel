#%%
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#%%
filename = "results/metrics.mat"
Acc = loadmat(filename)['Acc']
F1 = loadmat(filename)['F1']
Sens = loadmat(filename)['Sens']
Spec = loadmat(filename)['Spec']
Kappa = loadmat(filename)['Kappa']
# %%
l = ["Acc", "F1", "Sens", "Spec", "Kappa"]
labels = np.repeat(l, 10).tolist()
measures = np.squeeze(np.concatenate((Acc, F1, Sens, Spec, Kappa), axis=1)).tolist()
# %%
plot_dict = {"metrics":measures, "label":labels}
df = pd.DataFrame(plot_dict)
# %%
sns.boxplot(df, x = 'metrics', y = 'label')
# %%
