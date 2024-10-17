#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow as pa
import seaborn as sns
from datetime import timedelta, datetime

dir = Path(r"S:\Fackler_OSS_364376\data\IRB-364376-v1-230215")
# %%
fp = dir.joinpath('EHR', 'd_flo_measures.csv.gz')
dict = pd.read_csv(fp, compression="gzip")

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

ctr = 1
#%%
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
    limit = 5000

    if np.any(np.abs(patient_multi['SBS']) >= 2):
        y_data = patient_multi['SBS']
        X_data = patient_multi.drop(columns=['SBS'], axis = 1)
        j = 0
        l = X_data.shape[0]
        while l > 0:
            end = limit if l > limit else l
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(11, 8))
            labels = ['HR', 'PVC', 'RR', 'SPO2-%']

            for i, ax in enumerate(axs.flatten()):  
                color = 'tab:red'
                ax.set_xlabel('time (min)')
                ax.set_ylabel(labels[i], color=color)
                ax.margins(0.1, 0.1)
                ax.plot(X_data[labels[i]][:end].to_numpy(), 'r', label=labels[i])
                ax.tick_params(axis='y', labelcolor=color)

                ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:blue'
                ax2.set_ylabel('SBS', color=color)  # we already handled the x-label with ax1
                ax2.margins(0.1, 0.1)
                ax2.plot(y_data[:end].to_numpy(), 'bo', markersize = 5, label='SBS')
                ax2.tick_params(axis='y', labelcolor=color)      
                
                
                ax.set_title(labels[i])
            
            fn = "images_highres/patient" + str(ctr) + "_" + str(j) + ".png"
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            fig.savefig(fn)
            plt.close(fig)
            X_data = X_data[end:]  
            y_data = y_data[end:]
            j += 1
            l = X_data.shape[0]
        
    ctr += 1

# %%
