import os
import pandas as pd
from scipy.io import loadmat
import numpy as np

data_dir = os.path.join(os.path.dirname(__file__), 'data')
raw_data = loadmat(os.path.join(data_dir, 'Patient3_ECG.mat'))

keys = list(raw_data.keys())

for key in keys[3:len(keys)]:
    raw_data[key] = raw_data[key].squeeze()
    print(key)

sbs_score = raw_data['sbs_score']
start_time = raw_data['start_time']
end_time = raw_data['end_time']
ecg1 = raw_data['ecg1']
ecg2 = raw_data['ecg2']
ecg3 = raw_data['ecg3']

df = pd.DataFrame()

for i in range(len(sbs_score)):
    start_time[i] = start_time[i].squeeze()
    end_time[i] = end_time[i].squeeze()

    delta_t = (end_time[i] - start_time[i]) / len(ecg1[i])
    time = np.arange(start_time[i], end_time[i], delta_t)

    df_i = pd.DataFrame({
        'Time': time,
        'SBS_Score': sbs_score[i],
        'ECG_1': ecg1[i],
        'ECG_2': ecg2[i],
        'ECG_3': ecg3[i]
    })

    df = pd.concat([df, df_i], ignore_index=True)

print(df.shape)
print(df.head())

df.to_csv(os.path.join(data_dir, 'pat3.csv'), index=False)
print('Data saved to pat3.csv')