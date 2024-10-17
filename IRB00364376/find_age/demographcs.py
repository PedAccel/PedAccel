#%%
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd
import os

#%%
# read ptsd records
dir = Path(r"S:\Fackler_OSS_364376\data\IRB-364376-v1-230215")

fp = dir.joinpath('EHR', 'ptsd_record.csv.gz')
ptsd_record = pd.read_csv(fp, compression="gzip")
patients_ptsd = ptsd_record['pat_enc_csn_sid'].unique()

#%%
fp = dir.joinpath('EHR', 'flowsheet.csv.gz')
data = pd.read_csv(fp, compression="gzip")
data = data.drop(columns = ['meas_comment', 'meas_template_id'])
# flow_meas_id for SBS and RASS
fmid = [304080016, 304080017, 304080018, 304080019, 304080020, 304080021]
sbs = data[data['meas_id'].isin(fmid)]
sbs['SBS'] = sbs['meas_id'] - 304080019
sbs = sbs.drop(columns=['meas_value', 'meas_id'])
sbs['recorded_time'] = pd.to_datetime(sbs['recorded_time'], format='%Y-%m-%d %H:%M:%S')
sbs_indiv = sbs.groupby('pat_enc_csn_sid')

#%%
fp = dir.joinpath('EHR', 'vent_dur.csv.gz')
vent_record = pd.read_csv(fp, compression="gzip")
mech_vent = [5, 6, 7]
vent_record = vent_record[vent_record['level'].isin(mech_vent)]
patient_on_vent = set(vent_record['pat_enc_csn_sid'])
#%%
fp = dir.joinpath('EHR', 'patient.csv.gz')
gender_record = pd.read_csv(fp, compression="gzip")
gender = gender_record.groupby('osler_sid')['gender'].aggregate(lambda x: x.values[0])
gender_df = pd.DataFrame({'osler_sid': gender.index, "gender": gender.values})
#%%
fp = dir.joinpath('EHR', 'prism3.csv.gz')
prism = pd.read_csv(fp, compression='gzip')
ga_merge = pd.merge(prism, gender_df, on='osler_sid').groupby('pat_enc_csn_sid')
#%%
fp = dir.joinpath('EHR', 'med_admin.csv.gz')
med_admin = pd.read_csv(fp, compression="gzip")

excl_drugs = ['NEUROMUSCULAR BLOCKING AGENTS', 'BETA-ADRENERGIC AGENTS', 'ALPHA/BETA-ADRENERGIC BLOCKING AGENTS', 'BETA-ADRENERGIC BLOCKING AGENTS']

med_admin_filter = med_admin[med_admin['pharm_classname'].isin(excl_drugs)]
# med_admin_filter['pharm_classname'].unique()
patients_med = set(med_admin_filter['pat_enc_csn_sid'])
#%%
gender_age = ga_merge[['age_m', 'gender']].aggregate(lambda x: x.mean() if np.issubdtype(x, np.number) else x.mode().iloc[0])
#%%
bins = [10, 25, 50, 75, 100]
lab = ['10-25', '25-50', '50-75', '75-100']
#%%
pt_demographcs = gender_age.reset_index()

# %%
pt_demographcs['age_bin'] = pd.cut(pt_demographcs['age_m'], bins=bins, labels=lab)

# %%
pt_demographcs['is_valid'] = pt_demographcs['pat_enc_csn_sid'].apply(lambda x: (x in patients_ptsd) and (x in patients_ehr))
pt_demographcs = pt_demographcs[pt_demographcs['is_valid'] == True]
# %%
pt_demographcs['on_vent'] = pt_demographcs['pat_enc_csn_sid'].apply(lambda x: x in patient_on_vent)
# %%
not_on_med = pt_demographcs['pat_enc_csn_sid'].apply(lambda x: not(x in patients_med))
pt_demographcs = pt_demographcs[not_on_med]

#%%
extreme_values = sbs_indiv['SBS'].aggregate(lambda x: np.any(np.abs(x) >= 2))
is_extreme = pt_demographcs['pat_enc_csn_sid'].apply(lambda x: extreme_values[x])
pt_demographcs = pt_demographcs[is_extreme]
# %%
counts = pt_demographcs.drop(["pat_enc_csn_sid", 'age_m', 'is_valid'], axis=1).groupby('age_bin').value_counts()
counts.to_csv("demographics.csv")
# %%
