#%%

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow as pa
import seaborn as sns
from datetime import timedelta, datetime
import torch
import pandas as pd
import tsfel
#%%

win_size_a = [-5, -10,-15,-20,-25,-30,-60,-5,-10,-15,-20,-25,-30,-60]
win_size_b = [5, 10, 15, 20, 25, 30, 60, 1, 1, 1, 1, 1, 1, 1]
labels = ['HR', 'RR', 'SPO2-%']
lab = ['10-25', '25-50', '50-75', '75-100']

#%%
win_size = pd.read_csv('win_size.csv')
win_size = win_size.drop(columns=['Unnamed: 0'])
win_feat = win_size.iloc[:, 6:130]
#%%
super_folder = "images_feature/"
for bio_feat in labels:
    folder = super_folder + bio_feat.replace(' ', '') + '/'
    if not os.path.isdir(folder):
        os.mkdir(folder)
    corr = dict([(l, np.empty((len(win_size_a), win_feat.shape[1]))) for l in lab])
    for age_g in lab:
        for j, feat in enumerate(win_feat.columns):
            k = 0
            for a, b in zip(win_size_a, win_size_b):
                conditions = (win_size['label_data']==bio_feat) &\
                     (win_size['win_size_data'] == abs(a)) &\
                     (win_size['one_sided'] == (b==1)) &\
                     (win_size['age_bin'] == age_g)
                x_ = win_size[conditions][feat].to_numpy()
                y_ = win_size[conditions]['sbs_data'].to_numpy()
                corr[age_g][k][j] = np.corrcoef(x_, y_)[0][1]
                k += 1
    chuncks = list(range(0, len(win_feat.columns), 8)) + [len(win_feat.columns)]
    for j in range(len(chuncks) - 1):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(11, 11))

        for i, ax in enumerate(axs.flatten()): 
            sns.heatmap(corr[lab[i]][:, chuncks[j]:chuncks[j+1]], ax=ax, xticklabels=win_feat.columns[chuncks[j]:chuncks[j+1]], \
                    yticklabels=[str(a) + "_" +str(b) for a, b in zip(win_size_a, win_size_b)], \
                    annot=True, vmin=-1, vmax=1, cmap="vlag")  
            ax.set_title(lab[i])

        fn = folder + "heat_" + bio_feat.replace(' ', '')+str(j) + ".png"
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(fn)
        plt.close(fig)
# %%
