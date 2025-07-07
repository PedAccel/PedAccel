# Import Modules

import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from scipy.io import loadmat
import scipy
from scipy.stats import ttest_ind, skew, ttest_rel
import Actigraph_Metrics
from pygt3x.reader import FileReader 


data_dir = r'/Users/sidharthraghavan/Library/CloudStorage/OneDrive-Personal/Sid_stuff/PROJECTS/PedAccel/data_analysis/Vitals_accel_analysis/PatientData'
window_size = 100 #100 is 1 second worth of time
lead_time = 10
slice_size_min = 11
sr = .5

def load_gt3x_data(gt3x_filepath, to_numpy=False, verbose=False):
    '''
    Load data from GT3X file
    Expect data to have 3 columns (X, Y, Z) and a timestamp index
    '''
    with FileReader(gt3x_filepath) as reader:
        df = reader.to_pandas()
        df.reset_index(inplace=True)
        col_names = df.columns.values.tolist()
    if verbose:
        print(df.head())
        print(col_names)    
    if to_numpy:
        array = df.to_numpy()
        if verbose:
            print(array.shape)
        return array, col_names

    return df, col_names

def calculate_pre_sbs_stats(patient_dir, patient):
    """
    Calculate statistics from 12-hour window before each SBS score
    """
    # Load raw vitals data
    vitals_file = os.path.join(patient_dir, f'{patient}_SickBayData.mat')
    vitals_data = loadmat(vitals_file)
    
    # Extract timestamps and convert to datetime - FIX HERE
    time_data = vitals_data['time'][0].flatten()
    time_strings = [str(item[0]) for item in time_data]  # Convert to regular Python strings
    # Convert to datetime with proper format
    timestamps = pd.to_datetime(time_strings, format='%m/%d/%Y %I:%M:%S %p')
    
    # Create DataFrames for each metric with actual timestamps
    hr_df = pd.Series(vitals_data['heart_rate'].flatten(), index=timestamps)
    spo2_df = pd.Series(vitals_data['SpO2'].flatten(), index=timestamps)
    rr_df = pd.Series(vitals_data['respiratory_rate'].flatten(), index=timestamps)
    
    # Load raw accelerometry data
    accel_file = os.path.join(patient_dir, f'{patient}_AccelData.gt3x')
    acti_data, acti_names = load_gt3x_data(accel_file)
    acti_data['mag'] = np.linalg.norm(acti_data[['X', 'Y', 'Z']].values, axis=1)
    acti_data['dts'] = pd.to_datetime(acti_data['Timestamp'], unit='s')
    accel_series = acti_data.set_index('dts')['mag']
    
    # Load SBS data to get timestamps
    data_filepath = os.path.join(patient_dir, f'{patient}_SICKBAY_10MIN_1MIN_Retro.mat')
    sbs_data = loadmat(data_filepath)
    # Convert start times to datetime with proper format - FIX HERE TOO
    sbs_times = pd.to_datetime([str(ts[0]) for ts in sbs_data['start_time'].flatten()], format='%Y-%m-%dT%H:%M:%S')

    # Manually input the timestamp for analysis
    manual_timestamp = pd.to_datetime('1-17-2024 20:51:00', format='%m-%d-%Y %H:%M:%S')

    # Only keep the SBS time that matches the manual timestamp
    # Keep SBS scores within 15 minutes of the manual timestamp
    time_diff = (sbs_times - manual_timestamp).abs()
    sbs_times = sbs_times[time_diff <= pd.Timedelta(minutes=15)]
    if len(sbs_times) == 0:
        raise ValueError(f"No SBS score found within 15 minutes of {manual_timestamp}")
    
    # Calculate statistics for 12 hours before each SBS score
    window_stats = []
    for sbs_time in sbs_times:
        # Define 12-hour window before SBS score
        window_start = sbs_time - pd.Timedelta(hours=12)
        
        # Get data in this window
        hr_window = hr_df[window_start:sbs_time]
        spo2_window = spo2_df[window_start:sbs_time]
        rr_window = rr_df[window_start:sbs_time]
        accel_window = accel_series[window_start:sbs_time]
        
        # Calculate statistics
        stats = {
            'heart_rate': {'mean': hr_window.mean(), 'std': hr_window.std()},
            'SpO2': {'mean': spo2_window.mean(), 'std': spo2_window.std()},
            'respiratory_rate': {'mean': rr_window.mean(), 'std': rr_window.std()},
            'accelerometry': {'mean': accel_window.mean(), 'std': accel_window.std()}
        }
        window_stats.append(stats)
    
    return window_stats

def normalize_sbs_data(patient_dir, patient, window_stats):
    """
    Normalize the SBS data using the pre-calculated window statistics
    """
    # Load the SBS data
    data_filepath = os.path.join(patient_dir, f'{patient}_SICKBAY_10MIN_1MIN_Retro.mat')
    data = loadmat(data_filepath)
    
    # Extract data
    x_mag = data["x_mag"]
    SBS = data["sbs"].flatten()
    hr = data['heart_rate']
    SpO2 = data['SpO2']
    rr = data['respiratory_rate']
    
    # Normalize each window using its corresponding pre-SBS statistics
    normalized_data = {
        'heart_rate': [],
        'SpO2': [],
        'respiratory_rate': [],
        'accelerometry': []
    }
    
    for i in range(len(window_stats)):
        stats = window_stats[i]
        
        # Normalize the data
        normalized_data['heart_rate'].append((hr[i] - stats['heart_rate']['mean']) / stats['heart_rate']['std'])
        normalized_data['SpO2'].append((SpO2[i] - stats['SpO2']['mean']) / stats['SpO2']['std'])
        normalized_data['respiratory_rate'].append((rr[i] - stats['respiratory_rate']['mean']) / stats['respiratory_rate']['std'])
        normalized_data['accelerometry'].append((x_mag[i] - stats['accelerometry']['mean']) / stats['accelerometry']['std'])

    # Plot normalized data for each metric in the 11-minute period
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    metrics = ['heart_rate', 'SpO2', 'respiratory_rate', 'accelerometry']
    titles = ['Normalized Heart Rate', 'Normalized SpO2', 'Normalized Respiratory Rate', 'Normalized Accelerometry']

    for idx, metric in enumerate(metrics):
        # Each entry in normalized_data[metric] is a vector for the 11-minute window
        for window in normalized_data[metric]:
            axs[idx].plot(window, alpha=0.7)
        axs[idx].set_ylabel('Z-score')
        axs[idx].set_title(titles[idx])

    axs[-1].set_xlabel('Time (sample index in 11-min window)')
    plt.tight_layout()
    plt.show()
    
    return normalized_data, SBS


if __name__ == "__main__":

    patient = 'Patient9'
    data_dir = r'/Users/sidharthraghavan/Library/CloudStorage/OneDrive-Personal/Sid_stuff/PROJECTS/PedAccel/data_analysis/Vitals_accel_analysis/PatientData'
    window_size = 100 #100 is 1 second worth of time
    lead_time = 10
    slice_size_min = 11
    patient_dir = os.path.join(data_dir, patient)
    if not os.path.isdir(patient_dir):
        print(f"Patient directory {patient_dir} does not exist.")
        sys.exit(1)
    print(f"Processing patient: {patient}")
    # Calculate pre-SBS statistics
    window_stats = calculate_pre_sbs_stats(patient_dir, patient)
    print("Pre-SBS statistics calculated.")
    # Normalize the SBS data
    normalized_data, SBS = normalize_sbs_data(patient_dir, patient, window_stats)
    print("SBS data normalized.")