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

dir = Path(r"S:\Fackler_OSS_364376\data\IRB-364376-v1-230215")
# %%
fp = dir.joinpath('EHR', 'd_flo_measures.csv.gz')

names = ["State Behavioral Scale",
"-3 Unresponsive", 
"-2 Responsive to noxious stimuli", 
"-1 Responsive to gentle touch or voice",
"0 Awake and Able to calm",
"+1 Restless and difficult to calm",
"+2 Agitated",
"State Behavioral Scale (SBS)"]

# note: flowsheet record flow_meas_id as meas_id
# note: SBS score values are only stored in these fields

fmid = [304080016, 304080017, 304080018, 304080019, 304080020, 304080021]
# %%
# connect with feather file
fp = dir.joinpath('EHR', 'ptsd_record.csv.gz')

ptsd_record = pd.read_csv(fp, compression="gzip")

# load flow table of all patient EHR records
fp = dir.joinpath('EHR', 'flowsheet.csv.gz')
data = pd.read_csv(fp, compression="gzip")
data = data.drop(columns = ['meas_comment', 'meas_template_id'])
# Note: pandas took 50 seconds to load the table. Consider porting to PySpark RDD

sbs = data[data['meas_id'].isin(fmid)]
# print(sbs.shape)

# calculate sbs score from offset
sbs['SBS'] = sbs['meas_id'] - 304080019
sbs = sbs.drop(columns=['meas_value', 'meas_id'])
sbs['recorded_time'] = pd.to_datetime(sbs['recorded_time'], format='%Y-%m-%d %H:%M:%S')
sbs_indiv = sbs.groupby('pat_enc_csn_sid')
#%%
fp = dir.joinpath('EHR', 'adt.csv.gz')
prism = pd.read_csv(fp, compression='gzip')
# prism = prism.groupby('pat_enc_csn_sid')['age_m']

#%%

# load pre-selected patients from patient_inclexcl.ipynb
patients = np.load('../DONOTPUSH/patients_wodrugs.npy', allow_pickle=True)
# %%
fp_hl7m = dir.joinpath('ptsd-phi', 'vitals-hl7m', "003", '1000000003-2016-07-07-0.0166667-1-HL7M.feather')
fp_tsdb = dir.joinpath('ptsd-phi', 'vitals-tsdb', "106", '1000002106-2019-01-22-1-TSDB.feather')
fp_gevital = dir.joinpath('ptsd-phi', 'vitals-sb', "672", '1000002672-2020-10-01-1-GEVITAL.feather')
fp_medibus = dir.joinpath('ptsd-phi', 'vitals-sb', "672", '1000002672-2020-10-01-1-MEDIBUSVITAL.feather')

df = pd.read_feather(fp_hl7m, columns=None, use_threads=True, storage_options=None)
names_hl7m = df.columns.tolist()

# print("hl7m", names_hl7m)
df = pd.read_feather(fp_tsdb, columns=None, use_threads=True, storage_options=None)
names_tsdb = df.columns.tolist()
# print("tsdb", names_tsdb)

df = pd.read_feather(fp_gevital, columns=None, use_threads=True, storage_options=None)
names_gevitals = df.columns.tolist()
#%%
tsdb_order = ['dts', 'HR', 'RR_2344', 'SpO2_7874', "PVC"]
hl7m_order = ['dts', 'HR', 'RR', 'SPO2-%', 'PVC']
ge_order = ['dts','PARM_HR','PARM_RESP_RATE', 'PARM_SPO2_1', 'PARM_PVC']

metrics = set(hl7m_order + tsdb_order + ge_order)
dims = len(hl7m_order)

#%%
# turn off error message on .loc
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'

#%%
cfg = tsfel.get_features_by_domain()
cfg['spectral']['LPCC']['parameters']['n_coeff']=5
fs = 1/60
#%%
ctr = 1
win_size_a = [-5, -10,-15,-20,-25,-30,-60,-5,-10,-15,-20,-25,-30,-60]
win_size_b = [5, 10, 15, 20, 25, 30, 60, 1, 1, 1, 1, 1, 1, 1]
labels = ['HR', 'PVC', 'RR', 'SPO2-%']
all_win_size_dict = {'age_m':[], 'sbs_data':[], 'win_size_data':[], 'label_data':[], \
    'one_sided':[], 'all_features':pd.DataFrame([]), 'patient_id':[]}

for p in patients:
    if ctr % 50 == 0:
        print(f'[{ctr} / {len(patients)}]: {p}')
    
    files = ptsd_record[ptsd_record['pat_enc_csn_sid'] == p]
    files['start_time'] = pd.to_datetime(files['start_time'], format='%Y-%m-%d %H:%M:%S.%f')
    files['end_time'] = pd.to_datetime(files['end_time'], format='%Y-%m-%d %H:%M:%S.%f')
    files.sort_values('start_time')

    devices = files['device']
    filename = files['filename'] + '.feather'
    startime = files['start_time']

    dfs = []
    size = 0

    for (d, fn, t0) in zip(devices, filename, startime): 
        # drop 'MEDIBUSVITAL' since it is a ventilator (we dont want ventilated patients)
        if fn.endswith('MEDIBUSVITAL.feather'):
            continue
        # HL7M', 'TSDB', 'GEVITAL'
        if (d.endswith('HL7M') or d.endswith('TSDB')):
            fp_device = 'vitals-' + d.lower()
        else:
            fp_device = 'vitals-sb'
        fp_p = str(p)[-3:] # last 3 digit of pat_enc_csn_sid is the subfolder
        fp = dir.joinpath('ptsd-phi', fp_device, fp_p, fn)
        if (fp == None):
            print(fp, "does not exist")
            continue

        df = pd.read_feather(fp, columns=None, use_threads=True, storage_options=None)
        df = df.filter(metrics)
        # rearrange columns according to device
        if d.endswith('HL7M'):
            df = df.reindex(columns=hl7m_order)
        elif d.endswith('TSDB'):
            df = df.reindex(columns=tsdb_order)
        elif d.endswith('GEVITAL'):
            df = df.reindex(columns=ge_order)

        df.loc[:,'dts'] = pd.to_timedelta(df.loc[:,'dts'], unit='s')
        df.loc[:,'dts'] = df.loc[:,'dts'] + t0

        # standardize column names
        # time, heart rate, central venous pressure, resp rate and spo2-%
        df.columns = ['dts', 'HR', 'RR', 'SPO2-%', 'PVC']
        dfs.append(df)

    # resample to 1min intervals using median value
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html
    vitals = (pd.concat(dfs, axis=0)
              .set_index('dts')
              .sort_values(by=['dts'])
              .resample('1T').median()
    )
    # add SBS
    sbs_p = (sbs_indiv.get_group(p)
            .sort_values('recorded_time')
            .drop(columns=['osler_sid', 'pat_enc_csn_sid'])
            .set_index('recorded_time')
            .resample('1T').median()
    )
    pat_age = prism.get_group(p).mean()
    # preserve vitals timestamps while adding SBS where available
    patient_multi = pd.merge(left=vitals, right=sbs_p, left_index=True, right_index=True, how='left')
    # patient_multi = patient_multi.set_index(pd.Series(patient_multi.index).dt.round('T').astype('int64')//10**9//60)
    
    if np.any(np.abs(patient_multi['SBS']) >= 2):
        y_data = patient_multi['SBS']
        X_data = patient_multi.drop(columns=['SBS'], axis = 1)
        y_has_data = np.nonzero(y_data.isnull() == False)[0]

        win_size_dict = {'sbs_data':[], 'win_size_data':[], 'label_data':[], 'one_sided':[], \
            'patient_id':[], 'all_features':pd.DataFrame([]), 'age_m':[]}
        
        for a, b in zip(win_size_a, win_size_b):
            starts = []
            ends = []

            for idx in y_has_data:
                start = y_data.index[idx] + timedelta(seconds=60*a)
                end = y_data.index[idx] + timedelta(seconds=60*b)
                starts.append(start)
                ends.append(end)

            y_no_nan = y_data.dropna().tolist()
            for l in labels:
                index_ = 0
                for start, end in zip(starts, ends):
                    x_t = X_data[l][start:end]
                    y_t = y_data.iloc[y_has_data[index_]]
                    if x_t.isna().sum() == 0 and len(x_t)>5:
                        tsf = tsfel.time_series_features_extractor(cfg, x_t.to_numpy(), fs = fs, verbose = 0)
                        win_size_dict['all_features'] = pd.concat([win_size_dict['all_features'], tsf], ignore_index=True)
                        win_size_dict['win_size_data'].append(abs(a))
                        win_size_dict['label_data'].append(l)
                        win_size_dict['one_sided'].append(b == 1)
                        win_size_dict['patient_id'].append(ctr)
                        win_size_dict['age_m'].append(pat_age)
                        win_size_dict['sbs_data'].append(y_t)
                    index_ += 1
        
        for key in win_size_dict:
            if key != 'all_features':
                all_win_size_dict[key] += win_size_dict[key]
            else:
                all_win_size_dict[key] = pd.concat([all_win_size_dict[key], win_size_dict[key]], axis = 0, ignore_index=True)
        
    ctr += 1
#%%
all_win_size_dict['all_features'] = all_win_size_dict['all_features'].dropna(axis=1)
#%%
features_dict = all_win_size_dict['all_features'].to_dict(orient='list')
del all_win_size_dict['all_features']
all_win_size_dict.update(features_dict)
#%%
win_size = pd.DataFrame(all_win_size_dict)
win_size = win_size.dropna(axis=0)
#%%
bins = [10, 25, 50, 75, 100]
lab = ['10-25', '25-50', '50-75', '75-100']
win_size['age_bin'] = pd.cut(win_size['age_m'], bins=bins, labels=lab)
#%%
labels = ['HR', 'PVC', 'RR', 'SPO2-%']
# %%
super_folder = "images_feature/"
for feature in features_dict.keys():
    folder = super_folder + feature.replace(' ', '') + '/'
    if not os.path.isdir(folder):
        os.mkdir(folder)
    corr = dict([(l, np.empty((len(win_size_a), len(lab)))) for l in labels])
    k = 0
    for a, b in zip(win_size_a, win_size_b):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(15, 8))
        
        for i, ax in enumerate(axs.flatten()):  
            conditions = (win_size['label_data']==labels[i]) & (win_size['win_size_data'] == abs(a))& (win_size['one_sided'] == (b==1)) 
            sns.boxplot(legend = True, data=win_size[conditions], \
                x='sbs_data', y=feature, hue='age_bin', ax=ax, showfliers=False)
            ax.set_xlabel('SBS')
            ax.margins(0.1, 0.1)
            ax.set_ylabel(labels[i])  
            ax.set_title(labels[i])

            for m, age in enumerate(lab):
                conditions = (win_size['label_data']==labels[i]) & (win_size['win_size_data'] == abs(a)) & (win_size['one_sided'] == (b==1)) & (win_size['age_bin'] == age)
                x_ = win_size[conditions][feature].to_numpy()
                y_ = win_size[conditions]['sbs_data'].to_numpy()
                
                
                corr[labels[i]][k][m] = np.corrcoef(x_, y_)[0][1]
            
        k += 1
        
        fn = folder +'win'+ str(abs(a)) + "_" +str(abs(b)) + ".png"
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(fn)
        plt.close(fig)

    folder = "images_heat/"
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(11, 11))
        
    for i, ax in enumerate(axs.flatten()): 
        sns.heatmap(corr[labels[i]], ax=ax, xticklabels=lab, \
            yticklabels=[str(a) + "_" +str(b) for a, b in zip(win_size_a, win_size_b)], annot=True)  
        ax.set_title(labels[i])

    fn = folder + "heat_" + feature.replace(' ', '') + ".png"
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(fn)
    plt.close(fig)

# %%
