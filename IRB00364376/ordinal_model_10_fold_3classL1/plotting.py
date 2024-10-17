#%%
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os
#%%
plot_dict = {"metrics":[], "label":[], 'fold_id':[]}
y_preds = []
y_trues = []
test_ids = []
features = []
X_waves = []
l = ["Acc", "F1", "Sens", "Spec", "Kappa"]
#%%
dir = Path(r"S:\Fackler_OSS_364376\data\IRB-364376-v1-230215")
fp = dir.joinpath('EHR', 'adt_adm.csv.gz')
prism = pd.read_csv(fp, compression='gzip')
prism = prism.groupby('pat_enc_csn_sid')["hospital_service"]
#%%
for repeat in range(10):
    filename = f"results/cv{repeat}.mat"
    Acc = loadmat(filename)['Acc']
    F1 = loadmat(filename)['F1']
    Sens = loadmat(filename)['Sens']
    Spec = loadmat(filename)['Spec']
    Kappa = loadmat(filename)['Kappa']
    y_pred = loadmat(filename)['y_pred']
    y_true = loadmat(filename)['y_true']
    # X_wave = loadmat(filename)['X_wave']
    # test_id = loadmat(filename)['test_ids']
    importance =  loadmat(filename)['importance']
    labels = np.repeat(l, 10).tolist()
    measures = np.squeeze(np.concatenate((Acc, F1, Sens, Spec, Kappa), axis=1)).tolist()
    plot_dict['metrics'] += measures
    plot_dict['label'] += labels
    plot_dict['fold_id'] += [repeat] * len(labels)
    y_preds += np.squeeze(y_pred).tolist()
    y_trues += np.squeeze(y_true).tolist()
    # test_ids += np.squeeze(test_id).tolist()
    # X_waves.append(X_wave)
    features.append(importance)
#%%
X_waves =  np.concatenate(X_waves, axis = 0)
#%%
df = pd.DataFrame(plot_dict)
#%%
folder = "results/"
plot = sns.boxplot(df, x = 'label', y = 'metrics', hue='fold_id', legend=False)
fig = plot.get_figure()
fig.savefig(folder + "overall.png")

#%%
folder = "results/"
cnf = metrics.confusion_matrix(y_trues, y_preds)
cnf_pct = cnf / cnf.sum()

off_diag_mask = np.eye(*cnf_pct.shape, dtype=bool)
vmin = cnf_pct.min()
vmax = cnf_pct.max()

fig = plt.figure()
sns.heatmap(cnf_pct, annot=True, mask=~off_diag_mask, cmap="Blues", vmin=vmin, vmax=vmax)
sns.heatmap(cnf_pct, annot=True, mask=off_diag_mask, cmap="OrRd", vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]))
fig.savefig(folder + "confusion_matrix_percent.png")
plt.close(fig)

#%%
for act in range(0, 3):
    for pred in range(0, 3):
        idxs = np.argwhere((np.array(y_trues) == act) & (np.array(y_preds) == pred))
        idxs = np.random.choice(idxs.squeeze(), size = 20)
        image_folder_name = folder + f"pred{pred}act{act}/"
        if not os.path.isdir(image_folder_name):
            os.mkdir(image_folder_name)
        for i in idxs:
            pt_id = pt_id = test_ids[i]
            admission_department = (prism.get_group(pt_id).iloc[0]).replace(" ", "_")
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=False,figsize=(11, 8))
            labels = ['HR', 'RR', 'SPO2']
            ranges = [[50, 220], [0, 100], [50, 150]]
            for i_, ax in enumerate(axs.flatten()): 
                if i_ > 2:
                    break 
                color = 'tab:red'
                ax.set_xlabel('time (min)')
                ax.set_ylabel(labels[i_], color=color)
                ax.plot(X_waves[i, i_, :].squeeze(), 'r', label=labels[i_])
                ax.set_ylim(ranges[i_])
                ax.tick_params(axis='y', labelcolor=color)    
                ax.set_title(labels[i_] + " pred " + str(pred-1) + " actual " + str(act-1) + " age " + str(X_waves[i, 3, 0].squeeze().tolist()) + " department " + admission_department)

            fn = image_folder_name + "instance" + str(i) + ".png"
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            fig.savefig(fn)
            plt.close(fig)
#%%
folder = "results/"
cnf = metrics.confusion_matrix(y_trues, y_preds)
cnf_pct = cnf 
plot = sns.heatmap(cnf_pct, annot = True)
fig = plot.get_figure()
fig.savefig(folder + "confusion_matrix.png")
# %%
features = np.vstack(features).mean(axis = 0)
import json
f = open('feat.json')
feature_names = json.load(f)

# %%
feature_dict = {'importance': np.abs(features).tolist(), 'name':feature_names}
feature_df = pd.DataFrame(feature_dict)
feature_df = feature_df.sort_values(by='importance', ascending=False)
plot = sns.barplot(feature_df.head(20), x = 'name', y='importance')
plot.tick_params(axis='x', rotation=90)
fig = plot.get_figure()
fig.savefig(folder + "importance.png")
# %%
