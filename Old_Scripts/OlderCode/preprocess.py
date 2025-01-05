'''
Acceleration Preprocessing Code
* Requires GT3X File
* Requires SBS Excel Sheet OR Vitals Preprocessed MATLAB File
'''

import pandas as pd
import numpy as np
from pygt3x.reader import FileReader 
import os
from scipy.io import savemat
from functools import reduce
from scipy.io import loadmat
import preprocess_sickbay

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


def load_from_excel(sbs_filepath, to_numpy=False, verbose=False):
    # TODO: define a data model and ingest all the metadata we care about at once elegantly
    '''
    Load data from Excel file

    WARNING! This function assumes that 
    1. the header is on the 3rd row. 
    2. the excel contains a column 'SBS'
    This is very fragile. We should consider the regularity of our excel formatting. 
    '''
    df = pd.read_excel(sbs_filepath, header=0)
    col_names = df.columns.values.tolist()
    if 'SBS' not in col_names:
        raise ValueError('SBS column not found in the excel file')
    if verbose:
        print(df.head())
        print(col_names)
    if to_numpy:
        array = df.to_numpy()
        if verbose:
            print(array.shape)
        return array, col_names

    return df, col_names

def load_and_segment_data_excel(data_dir, window_size=10, lead_time=10):
    '''
    Load actigraphy and EPIC data from a directory and segment it into time windows.

    Assume that data_dir contains a directory for each patient, and all directories in data_dir are patient directories. Each patient directory must contain the actigraphy file and the EPIC file. 

    All patient files must be prefixed by their folder name. For example:
    Patient9
    |_Patient9_AccelData.gt3x
    |_Patient9__SBS_Scores.xlsx
    Patient11
    |_Patient11_AccelData.gt3x
    |_Patient11__SBS_Scores.xlsx
    '''
    # Search for patient directories in the data directory
    for patient in os.listdir(data_dir):
        # Filter out non-directories
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)

            print('Loading actigraphy data')
            actigraphy_filepath = os.path.join(patient_dir, patient + '_AccelData.gt3x')
            if not os.path.isfile(actigraphy_filepath):
                raise FileNotFoundError(f'Actigraphy file not found: {actigraphy_filepath}')
            acti_data, acti_names = load_gt3x_data(actigraphy_filepath)
            acti_data['mag'] = np.linalg.norm(acti_data[['X', 'Y', 'Z']].values, axis=1)
            acti_data['dts'] = pd.to_datetime(acti_data['Timestamp'], unit='s')
            print(acti_data.shape)
            print(acti_names)
        
            # SBS Scores from Excel File
            print('Loading SBS data')
            sbs_file = os.path.join(patient_dir, patient + '_SBS_Scores_Validated.xlsx')
            if not os.path.isfile(sbs_file):
                raise FileNotFoundError(f'Actigraphy file not found: {sbs_file}')
            epic_data, epic_names = load_from_excel(sbs_file)
            epic_data.dropna(subset=['SBS'], inplace = True) # drop rows with missing SBS scores
            print(epic_data.shape)
            print(epic_names)
            epic_data['dts'] = pd.to_datetime(epic_data['Time_uniform'], format='%m/%d/%Y %H:%M:%S %p')
            # precompute start and end time for each SBS recording
            epic_data['start_time'] = epic_data['dts'] - pd.Timedelta(lead_time, 'minutes')
            epic_data['end_time'] = epic_data['dts'] + pd.Timedelta(window_size - lead_time, 'minutes')

            print('Processing')
            windows = []
            sbs = []
            for i, row in epic_data.iterrows():
                # don't like the for-loop, but its not a big bottleneck for the number of SBS recordings we are getting right now. 

                in_window = acti_data[(acti_data['dts'] > row['start_time']) & (acti_data['dts'] < row['end_time'])].loc[:, ['dts', 'mag']]
                in_window.rename(columns={'mag': f'mag_{i}'}, inplace=True)
                if in_window.shape[0] > 0:
                    sbs.append(row['SBS'])
                    in_window['dts'] = in_window['dts'] - row['start_time']
                    windows.append(in_window)
                else:
                    print('No matching accelerometry data for SBS recording at ', row['dts'])

            print('Save to file')
            windows_merged = reduce(lambda  left,right: pd.merge(left,right,on=['dts'], how='outer'), windows)
            windows_merged.drop('dts', axis=1, inplace=True)
            windows_merged = windows_merged.apply(pd.to_numeric, downcast='float') #float32 is enough
            windows_merged.interpolate(axis=1, inplace=True) #fill na with linear interpolation

            x_mag = np.transpose(windows_merged.to_numpy())
            assert not np.isnan(np.sum(x_mag)) # fast nan check
            sbs = np.array(sbs)
            print(x_mag.shape)
            print(sbs.shape)

            filename = f'{patient}_{lead_time}MIN_{window_size - lead_time}MIN_Validated.mat'
            save_file = os.path.join(patient_dir, filename)
            savemat(save_file, dict([('x_mag', x_mag), ('sbs', sbs)]))


def load_and_segment_data_mat(data_dir, window_size=15, lead_time=10):
    '''
    Load actigraphy and vitals waveform MAT file from a directory and segment it into time windows. 
    '''
    # search for patient directories in the data directory
    for patient in os.listdir(data_dir):
        # filter out non-directories
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)

            print('Loading actigraphy data')
            actigraphy_filepath = os.path.join(patient_dir, patient + '_AccelData.gt3x')
            if not os.path.isfile(actigraphy_filepath):
                raise FileNotFoundError(f'Actigraphy file not found: {actigraphy_filepath}')
            acti_data, acti_names = load_gt3x_data(actigraphy_filepath)
            acti_data['mag'] = np.linalg.norm(acti_data[['X', 'Y', 'Z']].values, axis=1)
            acti_data['dts'] = pd.to_datetime(acti_data['Timestamp'], unit='s')
            print(acti_data.shape)
            print(acti_names)
            
            # SBS Scores from MAT File
            print('Loading SBS data')
            vitals_sbs_file = os.path.join(patient_dir, f'{patient}_SICKBAY_{lead_time}MIN_{window_size - lead_time}MIN_Validated_Default.mat')
            
            # Implement error handling here if file does not exist...
            vitals_data = loadmat(vitals_sbs_file)
            # print(vitals_data)
            SBS = vitals_data['sbs'].flatten()
            # Flatten the nested arrays
            start_time_flat = vitals_data['start_time'].flatten()
            end_time_flat = vitals_data['end_time'].flatten()

            # Convert the flattened arrays to Timestamp objects
            start_time = [pd.Timestamp(str(ts[0])) for ts in start_time_flat]
            end_time = [pd.Timestamp(str(ts[0])) for ts in end_time_flat]

            epic_data = pd.DataFrame({
                'SBS': SBS,
                'start_time': start_time,
                'end_time': end_time
            })
        
            print('Processing')
            windows = []
            sbs = []
            for i, row in epic_data.iterrows():
                # don't like the for-loop, but its not a big bottleneck for the number of SBS recordings we are getting right now. 

                in_window = acti_data[(acti_data['dts'] > row['start_time']) & (acti_data['dts'] < row['end_time'])].loc[:, ['dts', 'mag']]
                in_window.rename(columns={'mag': f'mag_{i}'}, inplace=True)
                if in_window.shape[0] > 0:
                    sbs.append(row['SBS'])
                    in_window['dts'] = in_window['dts'] - row['start_time']
                    windows.append(in_window)
                else:
                    print('No matching accelerometry data for SBS recording at ', row['dts'])

            print('Save to file')
            windows_merged = reduce(lambda  left,right: pd.merge(left,right,on=['dts'], how='outer'), windows)
            windows_merged.drop('dts', axis=1, inplace=True)
            windows_merged = windows_merged.apply(pd.to_numeric, downcast='float') #float32 is enough
            windows_merged.interpolate(axis=1, inplace=True) #fill na with linear interpolation

            x_mag = np.transpose(windows_merged.to_numpy())
            assert not np.isnan(np.sum(x_mag)) # fast nan check
            sbs = np.array(sbs)
            print(x_mag.shape)
            print(sbs.shape)
            
            # Vitals Data Preprocessed:
            hr = vitals_data['heart_rate']
            SpO2 = vitals_data['SpO2']
            rr = vitals_data['respiratory_rate']
            bps = vitals_data['blood_pressure_systolic']
            bpm = vitals_data['blood_pressure_mean']
            bpd = vitals_data['blood_pressure_diastolic']

            save_file = os.path.join(patient_dir, vitals_sbs_file)
            savemat(save_file, dict([('x_mag', x_mag), ('heart_rate', hr), 
                                     ('SpO2', SpO2), ('respiratory_rate', rr), ('blood_pressure_systolic', bps), 
                                     ('blood_pressure_mean', bpm), ('blood_pressure_diastolic', bpd), ('sbs', sbs)]))

if __name__ == '__main__':
    data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'
    # data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
    # load_and_segment_data(data_dir)
    load_and_segment_data_mat(data_dir)