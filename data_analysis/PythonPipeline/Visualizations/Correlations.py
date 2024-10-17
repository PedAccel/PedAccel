''''
Contains Functions to Calculate Correlations between Accelerometry, Vitals Data, and SBS
'''

# Import Packages
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
from pygt3x.reader import FileReader
import data_analysis.PythonPipeline.Data_Cleaning.Filtering as Filtering
import os
import sys
from scipy.stats import f_oneway, levene
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import tsfel
import math
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

vitals_names = ['hr', 'rr', 'spo2', 'bps', 'bpm', 'bpd']

def pearson_corr_accel_vitals(data_dir, lead_time):
    '''
    Calculates the pearson correlation coefficient between vitals and accelerometry waveform data for every patient
    '''

    # Iterate through each patient data folder
    for patient in os.listdir(data_dir):
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)
            
            print('Loading actigraphy data')
            actigraphy_filepath = os.path.join(patient_dir, f'{patient}_AccelData.gt3x')
            acti_data, acti_names = preprocess.load_gt3x_data(actigraphy_filepath)
            acti_data['mag'] = np.linalg.norm(acti_data[['X', 'Y', 'Z']].values, axis=1)
            acti_data['dts'] = pd.to_datetime(acti_data['Timestamp'], unit='s')
            
             # Load heart rate data
            print('Loading heart rate data')
            hr_data_mat = loadmat(os.path.join(patient_dir, f'{patient}_SickBayData.mat'))
            hr_data = pd.DataFrame({
                'timestamp': pd.to_datetime(hr_data_mat['time'][0]),
                'hr': hr_data_mat['heart_rate'][0]  # Adjust indexing as necessary
            })

            # Resample actigraphy data to the same rate as heart rate data
            print('Resampling data')
            freq = '2S'  # Example frequency; adjust as needed based on HR data rate
            acti_resampled = acti_data.resample(freq).mean().interpolate()
            hr_resampled = hr_data.resample(freq).mean().interpolate()
            
             # Ensure same length by trimming to the common timeframe
            common_start = max(acti_resampled.index.min(), hr_resampled.index.min())
            common_end = min(acti_resampled.index.max(), hr_resampled.index.max())
            acti_resampled = acti_resampled.loc[common_start:common_end]
            hr_resampled = hr_resampled.loc[common_start:common_end]

            # Calculate Pearson correlation coefficient
            correlation, _ = pearsonr(acti_resampled['mag'], hr_resampled['hr'])
            print(f'Pearson correlation coefficient for {patient}:', correlation)
# def pearson_corr_vitals_sbs(signal, sbs, signal_name, lead_time, slice_size_min):
#     fs = .5

#     # Generate configuration file for feature extraction
#     cfg_file = tsfel.get_features_by_domain()

#     # Extract features and restructure data
#     features_list = []
#     sbs_list = []
#     count = 0    
#     for i in range(len(signal)):
#         #signal = Actigraph_Metrics.VecMag_MAD(x_mag[i,:],100)
#         sbs_list.append(sbs[i])
#         features = tsfel.time_series_features_extractor(cfg_file, signal[i], fs, verbose=0)
#         features_list.append(features)
#         print(len(features_list[0]))

#         #list comprehension for column names
#         columns = [col for col in list(features_list[0])]
#         # Convert features and SBS scores to DataFrame
#         features_array = np.array(features_list).reshape(-1, len(columns)) #may need to change 389
#         df_features = pd.DataFrame(features_array)
#         df_features.columns = columns

#         #Pearson Correlation Coefficient
#         CCoeff = []
#         for i in columns:
#             y = sbs_list
#             myX = list(df_features[i])
#             nan_indices = [i for i, x in enumerate(myX) if math.isnan(x)]
#             myX = [x for x in myX if not math.isnan(x)]
#             cleaned_y = [val for idx, val in enumerate(y) if idx not in nan_indices]

#             corr, _ = pearsonr(cleaned_y, myX)
#             CCoeff.append(np.abs(corr))
#         my_dict = dict(zip(list(columns), list(CCoeff)))

#         # functional
#         clean_dict = filter(lambda k: not math.isnan(my_dict[k]), my_dict)
#         # dict comprehension
#         clean_dict = {k: my_dict[k] for k in my_dict if not math.isnan(my_dict[k])}

#         #Retrieve N features with best correlation coefficient  
#         # Initialize N
#         N = 3
            
#         # N largest values in dictionary
#         # Using sorted() + itemgetter() + items()
#         res = dict(sorted(clean_dict.items(), key=itemgetter(1), reverse=True)[:N])
        
#         #Plot a histogram
#         y = list(res.keys())
#         x = list(res.values()) #price
            
#         if len(x) != 0:
#             # Figure Size
#             fig, ax = plt.subplots(figsize =(10 ,5))
                
#             # Horizontal Bar Plot
#             ax.barh(y, x)
                
#             # Remove axes splines
#             for s in ['top', 'bottom', 'left', 'right']:
#                 ax.spines[s].set_visible(False)
                
#             # Remove x, y Ticks
#             ax.xaxis.set_ticks_position('none')
#             ax.yaxis.set_ticks_position('none')
                
#             # Add padding between axes and labels
#             ax.xaxis.set_tick_params(pad = 5)
#             ax.yaxis.set_tick_params(pad = 10)
                
#             # Add x, y gridlines
#             ax.grid(color ='grey',
#                     linestyle ='-.', linewidth = 0.5,
#                     alpha = 0.2)
                
#             # Show top values 
#             ax.invert_yaxis()

#             #set x axis range
#             ax.set_xlim([.8*min(x),1.1*max(x)])
            
#             # Add Plot Title
#             ax.set_title(f'Correlation between top features and SBS for\n {signal_name}_{lead_time}MIN_{slice_size_min - lead_time}MIN {signal_names[count]})',
#                         loc ='left', )
                
#             # Show Plot

#             plt.show()
#             count= count+1
            
#             return "The top N value pairs are " + str(res)
            
def pearson_corr_vitals_sbs(signal, sbs, signal_name, lead_time, slice_size_min):
    '''
    @param signal: vitals signal input
    @param sbs: sbs corresponding to vitals signal
    @param signal_name: name of input signal
    '''
    # Assuming tsfel and other necessary imports are already done

    cfg_file = tsfel.get_features_by_domain()
    features_list = []
    sbs_list = []
    fs = .5
    
    # Assuming signal and sbs are lists
    for i in range(len(signal)):
        sbs_list.append(sbs[i])
        features = tsfel.time_series_features_extractor(cfg_file, signal[i], fs, verbose=0)
        features_list.append(features)

    features_array = np.array(features_list).reshape(-1, 359)
    df_features = pd.DataFrame(features_array)
    
    #Pearson Correlation Coefficient
    my_dict = {}
    print(df_features.columns)
    columns = df_features.columns
    for i in columns:
        y = sbs_list
        x = list(df_features[i])
        if len(y) >= 2 and len(x) >= 2:
            corr, _ = pearsonr(y, x)
            my_dict[i] = np.abs(corr)
        else:
            my_dict[i] = np.nan

    # Filter out NaN values from the dictionary
    clean_dict = {k: v for k, v in my_dict.items() if not np.isnan(v)}

    # Retrieve N features with best correlation coefficient  
    # Initialize N
    N = 5
    
    # N largest values in dictionary
    # Using sorted() + itemgetter() + items()
    res = dict(sorted(clean_dict.items(), key=itemgetter(1), reverse=True)[:N])

    # Plotting
    y = list(res.keys())
    x = list(res.values())
    
    # Figure Size
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Horizontal Bar Plot
    ax.barh(y, x)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
    
    # Add x, y gridlines
    ax.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
    
    # Show top values 
    ax.invert_yaxis()
    
    # Set x axis range
    ax.set_xlim([0.8 * min(x), 1.1 * max(x)])
    
    # Add Plot Title
    ax.set_title(f'Correlation between top features and SBS for\n {signal_name}_{lead_time}MIN_{slice_size_min - lead_time}MIN)',
                            loc ='left', )
    
    # Show Plot
    plt.show()
    
    # Return the top N correlated features
    return "The top N value pairs are " + str(res)

def vitals_anova(data_dir, lead_time, slice_size_min):
    '''
    Calculates ANOVA between the 6 SBS groups and vitals data for each patient.
    @param lead_time is the time before each SBS score
    @param slice_size_min is the total time of each window
    '''

    sr = .5
    # Iterate through each patient data folder
    # There is no error handling in place, the .mat file must exist
    for patient in os.listdir(data_dir):
        # filter out non-directories
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)
            data_filepath_accel = os.path.join(patient_dir, f'{patient}_{lead_time}MIN_{slice_size_min - lead_time}MIN.mat')           
            data_filepath_vitals = os.path.join(patient_dir, f'{patient}_SICKBAY_{lead_time}MIN_{slice_size_min - lead_time}MIN.mat')

        accel_data = loadmat(data_filepath_accel)
        x_mag = accel_data["x_mag"]
        accel_SBS = accel_data["sbs"]
        
        vitals_data = loadmat(data_filepath_vitals)
        temp_hr = vitals_data['heart_rate']
        temp_SpO2 = vitals_data['SpO2']
        temp_rr = vitals_data['respiratory_rate']
        temp_bps = vitals_data['blood_pressure_systolic']
        temp_bpm = vitals_data['blood_pressure_mean']
        temp_bpd = vitals_data['blood_pressure_diastolic']
        vitals_SBS = vitals_data['sbs'].flatten()
        hr = []
        hr_sbs = []
        rr = []
        rr_sbs = []
        SpO2 = []
        SpO2_sbs = []
        bpm = []
        bpm_sbs = []
        bps = []
        bps_sbs = []
        bpd = []
        bpd_sbs = []

        print(f'Before: {temp_bpd.shape} for {patient}')

        # Filter the vitals data for each SBS score, create an SBS list for each vitals data
        for j in range(len(vitals_SBS)):
            if Filtering.checkVitals(temp_hr[j], slice_size_min, 'hr'):
                hr.append(temp_hr[j])
                hr_sbs.append(vitals_SBS[j])
            if Filtering.checkVitals(temp_rr[j], slice_size_min, 'rr'):
                rr.append(temp_hr[j])
                rr_sbs.append(vitals_SBS[j])
            if Filtering.checkVitals(temp_SpO2[j], slice_size_min, 'spo2'):
                SpO2.append(temp_hr[j])
                SpO2_sbs.append(vitals_SBS[j])
            if Filtering.checkVitals(temp_bps[j], slice_size_min, 'bps'):
                bpm.append(temp_hr[j])
                bpm_sbs.append(vitals_SBS[j])
            if Filtering.checkVitals(temp_bpm[j], slice_size_min, 'bpm'):
                bps.append(temp_hr[j])
                bps_sbs.append(vitals_SBS[j])
            if Filtering.checkVitals(temp_bpd[j], slice_size_min, 'bpd'):
                bpd.append(temp_hr[j])
                bpd_sbs.append(vitals_SBS[j])
        print(f'After: {np.array(bpd).shape} for {patient}')
        
        # Calculate the ANOVA Test Between Each Vital Data Set and SBS
        # Make a group for each SBS score
        for vital in vitals_names:
            sbs_neg3 = []
            sbs_neg2 = []
            sbs_neg1 = []
            sbs_0 = []
            sbs_1 = []
            sbs_2 = []
            count = 0
            for sbs in f'{vital}_sbs':
                if sbs == -3:
                    sbs_neg3.append(f'{vital[count]}')
                if sbs == -2:
                    sbs_neg2.append(f'{vital[count]}')
                if sbs == -1:
                    sbs_neg1.append(f'{vital[count]}')
                if sbs == 0:
                    sbs_0.append(f'{vital[count]}')
                if sbs == 1:
                    sbs_1.append(f'{vital[count]}')
                if sbs == 2:
                    sbs_2.append(f'{vital[count]}')
                count += 1
                
                print(sbs_0)
                # -------------
                # List to store groups with data
                groups_with_data = []
                
                # Check if each group has data and add them to the list
                if sbs_neg3:
                    groups_with_data.append(sbs_neg3)
                if sbs_neg2:
                    groups_with_data.append(sbs_neg2)
                if sbs_neg1:
                    groups_with_data.append(sbs_neg1)
                if sbs_0:
                    groups_with_data.append(sbs_0)
                if sbs_1:
                    groups_with_data.append(sbs_1)
                if sbs_2:
                    groups_with_data.append(sbs_2)
                
                # Perform Levene's test only if there is data in at least one group
                if groups_with_data:
                    statistic, p_value = levene(*groups_with_data)
                    print(f"Levene's test for {vital} p-value:", p_value)
        
        # statistic, p_value = levene(sbs_neg3, sbs_neg2, sbs_neg1, sbs_0, sbs_1, sbs_2)
        # print("Levene's test p-value:", p_value)

if __name__ == '__main__':
    # data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'
    data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'
    # data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
    # pearson_corr_accel_vitals(data_dir, 10)
    # pearson_corr_vitals_sbs(data_dir, 10, 15)
    vitals_anova(data_dir, 10, 15)