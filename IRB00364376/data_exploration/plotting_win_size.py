#%%

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow as pa
import seaborn as sns
from datetime import timedelta, datetime
import torch
import pandas as pd

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
ctr = 1
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

    # preserve vitals timestamps while adding SBS where available
    patient_multi = pd.merge(left=vitals, right=sbs_p, left_index=True, right_index=True, how='left')
    patient_multi = patient_multi.set_index(pd.Series(patient_multi.index).dt.round('T').astype('int64')//10**9//60)
    
    j = 0
    if np.any(np.abs(patient_multi['SBS']) >= 2):
        y_data = patient_multi['SBS']
        X_data = patient_multi.drop(columns=['SBS'], axis = 1)
        y_has_data = np.nonzero(y_data.isnull() == False)[0]

        win_size_a = [-5, -10,-15,-20,-25,-30,-60,-5,-10,-15,-20,-25,-30,-60]
        win_size_b = [5, 10, 15, 20, 25, 30, 60, 1, 1, 1, 1, 1, 1, 1]
        labels = ['HR', 'PVC', 'RR', 'SPO2-%']

        win_size_dict = {'x_data':[], 'sbs_data':[], 'win_size_data':[], 'label_data':[], 'one_sided':[]}
        
        for a, b in zip(win_size_a, win_size_b):
            starts = []
            ends = []
            for index in y_has_data:
                starts.append(index + a)
                ends.append(index + b)

            for l in labels:
                win_size_dict['sbs_data'] += (y_data.dropna().tolist())
                for start, end in zip(starts, ends):
                    start = 0 if start < 0 else start
                    end = X_data.shape[0] if end > X_data.shape[0] else end
                    x_t = X_data[l][start:end]
                    win_size_dict['x_data'].append(np.max(x_t)-np.min(x_t))
                    win_size_dict['win_size_data'].append(abs(a))
                    win_size_dict['label_data'].append(l)
                    win_size_dict['one_sided'].append(b == 1)
        
        win_size = pd.DataFrame(win_size_dict)
        win_size = win_size.dropna(axis=0)

        corr = dict([(l, np.empty((len(win_size_a)//2, 2))) for l in labels])
        
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(11, 8))
        
        for i, ax in enumerate(axs.flatten()):  
            sns.scatterplot(legend = False, data=win_size[win_size['label_data']==labels[i]], \
                x='sbs_data', y='x_data', hue='one_sided', size='win_size_data',ax=ax, alpha=0.5, marker='$\circ$')
            ax.set_xlabel('SBS')
            ax.margins(0.1, 0.1)
            ax.set_ylabel(labels[i])  # we already handled the x-label with ax1
            ax.set_title(labels[i])
            k = 0
            for a, b in zip(win_size_a, win_size_b):
                conditions = (win_size['label_data']==labels[i]) & (win_size['win_size_data'] == abs(a)) & (win_size['one_sided'] == (b==1))
                x_ = win_size[conditions]['x_data'].to_numpy()
                y_ = win_size[conditions]['sbs_data'].to_numpy()
                
                if not b == 1:
                    corr[labels[i]][k][0] = np.corrcoef(x_, y_)[0][1]
                else:
                    corr[labels[i]][k][1] = np.corrcoef(x_, y_)[0][1]
                
                k += 1
                k = k % (len(win_size_a)//2)
        
        fn = "images_win_size/patient" + str(ctr) + "_" + str(j) + ".png"
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(fn)
        plt.close(fig)

        j += 1 

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(11, 8))
        
        for i, ax in enumerate(axs.flatten()): 
            sns.heatmap(corr[labels[i]], ax=ax, xticklabels=['double', 'single'], \
                yticklabels=win_size_b[:corr[labels[i]].shape[0]], annot=True)  
            ax.set_title(labels[i])
        
        fn = "images_win_size/patient" + str(ctr) + "_" + str(j) + ".png"
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(fn)
        plt.close(fig)

    ctr += 1
    

    # %%
