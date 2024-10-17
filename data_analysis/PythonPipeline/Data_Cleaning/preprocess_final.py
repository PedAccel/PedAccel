'''
Acceleration and Vitals Preprocessing Code
|_ Loads Vitals and Accelerometry Data from GT3X and Excel Files and concatenates them with SBS Scoring Files.
|_ Outputs PatientX_SICKBAY_XMIN_YMIN.mat file
'''

# Import Modules
import pandas as pd
import numpy as np
from pygt3x.reader import FileReader 
import os
from scipy.io import savemat
from functools import reduce
from scipy.io import loadmat
import Filtering

# Define Variables
heart_rate = []
SpO2 = []
respiratory_rate = []
blood_pressure_systolic = []
blood_pressure_mean = []
blood_pressure_diastolic = []

vitals_list = [heart_rate, SpO2, respiratory_rate, blood_pressure_systolic, blood_pressure_mean,blood_pressure_diastolic]
names = ['heart_rate', 'SpO2', 'respiratory_rate', 'blood_pressure_systolic', 'blood_pressure_mean', 'blood_pressure_diastolic']

def load_from_excel(sbs_filepath, to_numpy=False, verbose=False):
    # Load data from Excel file
    df = pd.read_excel(sbs_filepath, header=0)
    col_names = df.columns.values.tolist()
    if 'SBS' not in col_names:
        raise ValueError('SBS column not found in the excel file')
    if to_numpy:
        array = df.to_numpy()
        return array, col_names
    return df, col_names

def load_segment_sickbay(data_dir, window_size=15, lead_time=10, tag = ""):
    '''
    Processes Sickbay Vitals MATLAB file and SBS Score Excel File
    * {patient}_SickBayData.mat (obtained from SickBayExtraction.py)
    * {patient}_SBS_Scores.xlsx (provided by CCDA_Extraction_SBS.py)
    '''
    sr = 0.5 # Sampling Rate
    # Iterate through patient directories
    for patient in os.listdir(data_dir):
        for i in vitals_list:
            i.clear()  # Clears each list in-place

        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing Sickbay:', patient)

            # Load SBS data
            sbs_file = os.path.join(patient_dir, f'{patient}_SBS_Scores.xlsx')
            if not os.path.isfile(sbs_file):
                raise FileNotFoundError(f'EPIC file not found: {sbs_file}')
            epic_data, epic_names = load_from_excel(sbs_file)
            
            # Statement to exclude SBS scores without stimulation
            # epic_data = epic_data[epic_data['Stim?'] == 'Y']
            
            # Statement for Default SBS Score Processing (Score 4)
            # for i in range(len(epic_data['SBS'])):
            #     if epic_data['Default?'][i] == 'Y':
            #         epic_data['SBS'][i] = 4
            
            epic_data.dropna(subset=['SBS'], inplace=True)
            epic_data['dts'] = pd.to_datetime(epic_data['Time_uniform'], format='mixed')
            epic_data['start_time'] = epic_data['dts'] - pd.Timedelta(lead_time, 'minutes')
            epic_data['end_time'] = epic_data['dts'] + pd.Timedelta(window_size - lead_time, 'minutes')
            
            # Load heart rate data
            vitals_file = os.path.join(patient_dir, f'{patient}_SickBayData.mat')
            if not os.path.isfile(vitals_file):
                raise FileNotFoundError(f'Heart rate file not found: {vitals_file}')
            vitals_data = loadmat(vitals_file)
            time_data = vitals_data['time'][0].flatten()  # Flatten nested array
            time_strings = [item[0] for item in time_data]  # Extract datetime strings

            # Convert datetime strings to datetime objects
            vitals_data['dts'] = pd.to_datetime([str(item) for item in time_strings], format='mixed')
            vitals_data['heart_rate'] = vitals_data['heart_rate'].flatten()  # Flatten heart rate array
            vitals_data['SpO2'] = vitals_data['SpO2'].flatten()  # Flatten heart rate array
            vitals_data['respiratory_rate'] = vitals_data['respiratory_rate'].flatten()  # Flatten heart rate array
            vitals_data['blood_pressure_systolic'] = vitals_data['blood_pressure_systolic'].flatten()  # Flatten heart rate array
            vitals_data['blood_pressure_mean'] = vitals_data['blood_pressure_mean'].flatten()  # Flatten heart rate array
            vitals_data['blood_pressure_diastolic'] = vitals_data['blood_pressure_diastolic'].flatten()  # Flatten heart rate array

            # Create a DataFrame from the dictionary
            vitals_data_df = pd.DataFrame({'dts': vitals_data['dts'], 'heart_rate': vitals_data['heart_rate'], 'SpO2': vitals_data['SpO2'], 'respiratory_rate': vitals_data['respiratory_rate']
                                           , 'blood_pressure_systolic': vitals_data['blood_pressure_systolic'], 'blood_pressure_mean': vitals_data['blood_pressure_mean']
                                           , 'blood_pressure_diastolic': vitals_data['blood_pressure_diastolic']})
            sbs = []
            
            #Time Variables
            start_time = []
            end_time = []

            for i, row in epic_data.iterrows():
                # Define the time window
                start_time_cur = row['start_time']
                end_time_cur = row['end_time'] 

                # Filter data within the time window
                in_window = vitals_data_df[(vitals_data_df['dts'] >= start_time_cur) & (vitals_data_df['dts'] <= end_time_cur)]
        
                if not in_window.empty:  # Check if any data values are found in the window
                    sbs.append(row['SBS'])
                    start_time.append(start_time_cur)
                    end_time.append(end_time_cur)

                    # Calculate the relative time within the window
                    in_window['dts'] = in_window['dts'] - row['start_time']

                    index = 0
                    for vital in vitals_list:
                        column = names[index]
                        temp_list = in_window[column].tolist()
                        vital.append(temp_list)
                        index+=1

            # Save Start/End times in correct format
            start_time_str = [ts.isoformat() for ts in start_time]
            end_time_str = [ts.isoformat() for ts in end_time]

            # Convert sbs to a numpy array
            sbs = np.array(sbs)
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
            
            # Further processing and saving...
            print('Save to file')

            # Remove empty lists from vitals_list and corresponding elements from names
            vitals_list_filtered = [v for v, n in zip(vitals_list, names) if v]
            names_filtered = [n for v, n in zip(vitals_list, names) if v]

            filename = f'{patient}_SICKBAY_{lead_time}MIN_{window_size-lead_time}MIN{tag}.mat'
            save_file = os.path.join(patient_dir, filename)
            filtered_dict = {name: vitals for name, vitals in zip(names_filtered, vitals_list_filtered)}

            # Filtering so that data is saved properly
            for i in range(len(vitals_list)):
                name = names[i]
                cur_list = filtered_dict[name] # cur_list is 2D
                for j in range(len(cur_list)):
                    cur_list[j] = np.array(cur_list[j]) #convert each sublist to an np array

                    # Sampling vitals in data has glitches where extra or not enough data is recorded.
                    # To compensate, we remove or fill values: 
                    print(f'before sampling: {len(cur_list[j])}')
                    expected_samples = window_size * 30 # Time(min) * 60 sec/min * sr(1sample/2 sec)
                    if(len(cur_list[j]) > expected_samples):
                        cut = len(cur_list[j])-expected_samples
                        cur_list[j] = cur_list[j][cut:]

                    elif(len(cur_list[j]) < expected_samples): # Linear extrapolation to make all subarrays the same length
                        # Append NaN values to the end of the list
                        num_missing_samples = expected_samples - len(cur_list[j])
                        nan_values = np.full(num_missing_samples, np.nan)
                        cur_list[j] = np.concatenate((cur_list[j], nan_values))
                        print(f'after sampling: {len(cur_list[j])}')
                cur_list = np.array(cur_list, np.dtype('float16')) # Save List of np arrays as an np array
            
            filtered_dict['sbs'] = np.array(sbs)
            # print(filtered_dict['start_time'])
            # savemat(save_file, filtered_dict, appendmat = False)
            
            # ADD IN Filtering Code Here...
            temp_hr = filtered_dict['heart_rate']
            temp_SpO2 = filtered_dict['SpO2']
            temp_rr = filtered_dict['respiratory_rate']
            temp_bps = filtered_dict['blood_pressure_systolic']
            temp_bpm = filtered_dict['blood_pressure_mean']
            temp_bpd = filtered_dict['blood_pressure_diastolic']
            vitals_SBS = filtered_dict['sbs'].flatten()
            hr = []
            rr = []
            SpO2 = []
            bpm = []
            bps = []
            bpd = []
            vitals_list_final = [hr,rr,SpO2,bpm,bps,bpd]
            vitals_names_final = ['hr','rr','spo2','bpm','bps','bpd']
            temp_vitals = [temp_hr,temp_rr, temp_SpO2,temp_bpm,temp_bps,temp_bpd] 
            
            # Generate a list to insert in place of invalid data, 
            # This list serves as a flag for a window to ignore in the box plot function
            flag_list = [0] * (int)(sr * 60 * window_size) 
            
            # Iterate through each SBS score for every vitals metric, assess validity of data
            for j in range(len(vitals_list_final)):
                print(f'original {vitals_names_final[j]} vitals array shape: {np.array(temp_vitals[j]).shape} ')
                for i in range(len(vitals_SBS)):
                    if (Filtering.checkVitals(temp_vitals[j][i], window_size, vitals_names_final[j])): # Check the data in a single window
                        vitals_list_final[j].append(temp_vitals[j][i]) # Append that single window data to the 2D hr, rr, spo2, bpm, bps, bpd arrays if that window's data is valid
                    else:
                        vitals_list_final[j].append(flag_list) # Append an array of zeros for window number i for the jth vitals metric if the data is invalid (i.e. too many NaN points)
                        print(f'{vitals_names_final[j]} SBS index {i} has insufficient data, zeros appended in place') 
                print(f'final {vitals_names_final[j]} vitals array shape: {np.array(vitals_list_final[j]).shape}') # The number of SBS scores by the number of samples in a window
            
            vitals_list_filtered_final = [v for v, n in zip(vitals_list_final, vitals_names_final) if v]
            names_filtered_final = [n for v, n in zip(vitals_list_final, vitals_names_final) if v]
            filtered_dict_final = {name: vitals for name, vitals in zip(names_filtered_final, vitals_list_filtered_final)}

            filtered_dict_final['start_time'] = np.array(start_time_str, dtype=object)
            filtered_dict_final['end_time'] = np.array(end_time_str, dtype=object)
            filtered_dict_final['sbs'] = np.array(vitals_SBS)
            savemat(save_file, filtered_dict_final, appendmat = False)

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

def load_and_segment_data_mat(data_dir, window_size=15, lead_time=15, tag = ""):
    '''
    Load actigraphy and vitals waveform MAT file from a directory and segment it into time windows. 
    PatientX
    |_PatientX_AccelData.gt3x
    |_PatientX_SickBayData.mat
    '''
    # load_segment_sickbay(data_dir, window_size, lead_time, tag)
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
            vitals_sbs_file = os.path.join(patient_dir, f'{patient}_SICKBAY_{lead_time}MIN_{window_size - lead_time}MIN{tag}.mat')
            
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
            # print(acti_data['dts'].head())
            print(acti_data.columns)
            windows = []
            sbs = []
            
            hr = vitals_data['hr']
            SpO2 = vitals_data['spo2']
            rr = vitals_data['rr']
            bps = vitals_data['bps']
            bpm = vitals_data['bpm']
            bpd = vitals_data['bpd']
            
            # Create new start_time/end_time variables that are for acti_data
            matched_start_times = []
            matched_end_times = []
            
            # Iterate through every SBS score in epic_data
            for i, row in epic_data.iterrows():
                # don't like the for-loop, but its not a big bottleneck for the number of SBS recordings we are getting right now. 

                in_window = acti_data[(acti_data['dts'] > row['start_time']) & (acti_data['dts'] < row['end_time'])].loc[:, ['dts', 'mag']]
                in_window.rename(columns={'mag': f'mag_{i}'}, inplace=True)
                if in_window.shape[0] > 0:
                    sbs.append(row['SBS'])
                    in_window['dts'] = in_window['dts'] - row['start_time']
                    windows.append(in_window)
                    matched_start_times.append(row['start_time'])
                    matched_end_times.append(row['end_time'])
                else:
                    # vitals_df.drop(index=i, inplace=True)
                    hr[i, :] = np.nan
                    SpO2[i, :] = np.nan
                    rr[i, :] = np.nan
                    bps[i, :] = np.nan
                    bpm[i, :] = np.nan
                    bpd[i, :] = np.nan
                    print('No matching accelerometry data for SBS recording at start time', row['start_time']) 
            hr = hr[~np.isnan(hr).any(axis=1)]
            rr = rr[~np.isnan(rr).any(axis=1)]
            SpO2 = SpO2[~np.isnan(SpO2).any(axis=1)]
            bpm = bpm[~np.isnan(bpm).any(axis=1)]
            bps = bps[~np.isnan(bps).any(axis=1)]
            bpm = bpm[~np.isnan(bpm).any(axis=1)]
            
            print('Save to file')
            windows_merged = reduce(lambda  left,right: pd.merge(left,right,on=['dts'], how='outer'), windows)
            windows_merged.drop('dts', axis=1, inplace=True)
            windows_merged = windows_merged.apply(pd.to_numeric, downcast='float') #float32 is enough
            windows_merged.interpolate(axis=1, limit_direction='both', inplace=True) #fill na with linear interpolation

            x_mag = np.transpose(windows_merged.to_numpy())
            assert not np.isnan(np.sum(x_mag)) # fast nan check
            sbs = np.array(sbs)
            print(x_mag.shape)
            print(sbs.shape)

            # # Vitals Data Preprocessed:
            # hr = vitals_df['hr']
            # SpO2 = vitals_df['spo2']
            # rr = vitals_df['rr']
            # bps = vitals_df['bps']
            # bpm = vitals_df['bpm']
            # bpd = vitals_df['bpd']
            # # final_file = os.path.join(patient_dir, f'{patient}_FULLDATA_{lead_time}MIN_{window_size - lead_time}MIN{tag}.mat')

            save_file = os.path.join(patient_dir, vitals_sbs_file)
            savemat(save_file, dict([('x_mag', x_mag), ('heart_rate', hr), 
                                     ('SpO2', SpO2), ('respiratory_rate', rr), ('blood_pressure_systolic', bps), 
                                     ('blood_pressure_mean', bpm), ('blood_pressure_diastolic', bpd), ('sbs', sbs)]))

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
        
            # SBS Scores from Excel File
            print('Loading SBS data')
            sbs_file = os.path.join(patient_dir, patient + '_SBS_Scores.xlsx')
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

if __name__ == '__main__':
    '''
    Set the following:
    |_ data_dir: current working directory
    |_ window_size_in: total window used in analysis
    |_ lead_time_in: length of analysis before SBS score
    |_ tag: string tag of mat file
        E.g., _Validated, _Nurse, _WSTIM, etc.
    '''
    data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'
    # data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
    window_size_in = 16
    lead_time_in = 15
    tag = "Nurse"
    #load_segment_sickbay(data_dir, window_size_in, lead_time_in, tag)
    load_and_segment_data_mat(data_dir, window_size_in, lead_time_in, tag)