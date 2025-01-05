'''
Regression.py
* Takes in the acceleration, vitals and finds correlation to SBS data.
* Acceleration and vitals data are processed by variance for the relevant score.
'''

#%%
# Import Modules
import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy.io import loadmat
import data_analysis.PythonPipeline.Data_Cleaning.Filtering as Filtering

#%%
# Load Data
data_filepath_accel = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData\Patient2\Patient2_10MIN_5MIN_Validated.mat'      
data_filepath_vitals = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData\Patient2\Patient2_SICKBAY_10MIN_5MIN_Validated_Stim.mat'
slice_size_min = 15
sr = 0.5

accel_data = loadmat(data_filepath_accel)
x_mag = accel_data["x_mag"]
SBS = accel_data["sbs"].flatten()

vitals_data = loadmat(data_filepath_vitals)
temp_hr = vitals_data['heart_rate']
temp_SpO2 = vitals_data['SpO2']
temp_rr = vitals_data['respiratory_rate']
temp_bps = vitals_data['blood_pressure_systolic']
temp_bpm = vitals_data['blood_pressure_mean']
temp_bpd = vitals_data['blood_pressure_diastolic']
vitals_SBS = vitals_data['sbs'].flatten()
hr = []
rr = []
SpO2 = []
bpm = []
bps = []
bpd = []
vitals_list = [hr,rr,SpO2,bpm,bps,bpd]
vitals_names = ['hr','rr','spo2','bpm','bps','bpd']
temp_vitals = [temp_hr,temp_rr, temp_SpO2,temp_bpm,temp_bps,temp_bpd] 
                
flag_list = [0] * (int)(sr * 60 * slice_size_min) #generate a list to insert in place of invalid data, 
# this list serves as a flag for a window to ignore in the box plot function

for j in range(len(vitals_list)): #go through every vitals metric
    print(f'original {vitals_names[j]} vitals array shape: {np.array(temp_vitals[j]).shape} ')
    for i in range(len(vitals_SBS)): #go through every SBS score for each vitals metric
        if (Filtering.checkVitals(temp_vitals[j][i], slice_size_min, vitals_names[j])): #check the data in a single window
            vitals_list[j].append(temp_vitals[j][i]) #append that single window data to the 2D hr,rr,spo2,bpm,bps,bpd arrays if that window's data is valid
        else:
            vitals_list[j].append(flag_list) #append an array of zeros for window number i for the jth vitals metric if the data is invalid(i.e. too many NaN points)
            print(f'{vitals_names[j]} SBS index {i} has insufficient data, zeros appended in place') 
    print(f'final {vitals_names[j]} vitals array shape: {np.array(vitals_list[j]).shape}') #should be the number of SBS scores by the number of samples in a window
# %%
# print(x_mag)
print(hr[38])

accel_var = []
hr_var = []
rr_var = []

for i in range(len(SBS)):
    accel_var.append(np.var(x_mag[i]))
    hr_var.append(np.var(hr[i]))
    rr_var.append(np.var(rr[i]))

X = accel_var, hr_var, rr_var
y = SBS

#%%
# Create a Linear Regression model
regr = linear_model.LinearRegression()
regr.fit(X, y)

# %%
