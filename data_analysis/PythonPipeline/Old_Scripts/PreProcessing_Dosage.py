#%%
import pandas as pd
import numpy as np
import os
from scipy.io import savemat
import sys
import sysconfig;
#%%
x_data = pd.read_csv('./Patient9_FullSBS.csv')
#x_data = pd.read_csv('./Users/jakes/Documents/DT 6 Analysis/PythonCode/Patient9_Data_Set1')
#%%
bolus_dose = pd.read_excel('./Patient_9_Dosage.xlsx', header=2, usecols='A:C')
#%%
bolus_dose['dts'] = pd.to_datetime(bolus_dose['Time_uniform'], format='%m/%d/%Y %H:%M:%S %p')
x_data['dts'] = pd.to_datetime(x_data['time'], format='mixed')
#%%
acc = x_data[['X', 'Y', 'Z']]
mag = np.linalg.norm(acc, axis=1)
x_data['mag'] = mag
#%%
bolus_dose = bolus_dose.dropna(axis = 0)
#%%
x_ = []
y = []
for index, row in bolus_dose.iterrows():
    dosage_time = row['dts']
    dosage_time_start = dosage_time - pd.Timedelta(5, 'minutes')
    dosage_time_end = dosage_time + pd.Timedelta(0, 'minutes')
    conditions = (x_data['dts'] > dosage_time_start) & (x_data['dts'] < dosage_time_end)
    x_mag = x_data[conditions]['mag'].to_numpy()
    if x_mag.shape[0] > 0:
        x_.append(x_mag[:59899]) #min length of all x_mag
        y.append(row['Midazolam Bolus (mg)'])
#%%
x_mag_data = np.vstack(x_)
dosage_data = np.array(y)
# %%
os.chdir(r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData\Patient9')
# os.chdir(r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\Data Analysis\PythonPipeline\PatientData\Patient9')
savemat('Patient9_BolusDosage.mat', dict([('x_mag', x_mag_data), ('dosage', dosage_data)]))
# %%