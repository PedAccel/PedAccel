'''
Acceleration and Vitals Preprocessing Code
|_ Loads Vitals and Accelerometry Data from .mat and .gt3x files, and concatenates them with SBS Scoring files (.xlsx).
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
import filtering
import gc


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

import pandas as pd



def load_segment_sickbay(data_dir, window_size=15, lead_time=10, tag = "Nurse"):
    '''
    Processes Sickbay Vitals MATLAB file and SBS Score Excel File
    * {patient}_SickBayData.mat (obtained from SickBayExtraction.py)
    * {patient}_SBS_Scores_{Tag}.xlsx (provided by CCDA_Extraction_SBS.py). Tag = "Nurse" or Tag = "Retro"
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
            # sbs_file = os.path.join(patient_dir, f'{patient}_SBS_Scores.xlsx')

            # SBS Scores from Excel File
            print('Loading SBS data')
            s = f"_SBS_Scores_{tag}.xlsx"
            sbs_file = os.path.join(patient_dir, patient + s)
            if not os.path.isfile(sbs_file):
                # raise FileNotFoundError(f'Actigraphy file not found: {sbs_file}')
                print("SBS File not found")
                continue 

            epic_data, epic_names = load_from_excel(sbs_file)
            
            # Statement to load Retrospective SBS Scores
            if(tag == "Retro"):
                epic_data = epic_data[(epic_data['SBS'] != '') & (epic_data['SBS'] != 'TODO')] # (epic_data['Default'] != 'Y') & 
            
            # Print the number of rows before dropping NaN values
            print(f"Number of rows before dropping NaN: {epic_data.shape[0]}")

            # Drop rows with NaN in the 'SBS' column
            epic_data.dropna(subset=['SBS'], inplace=True)

            # Print the number of rows after dropping NaN values
            print(f"Number of rows after dropping NaN: {epic_data.shape[0]}")

            epic_data['dts'] = pd.to_datetime(epic_data['Time_uniform'], format='mixed')
            epic_data['start_time'] = epic_data['dts'] - pd.Timedelta(lead_time, 'minutes')
            epic_data['end_time'] = epic_data['dts'] + pd.Timedelta(window_size - lead_time, 'minutes')
            
            print(epic_data.head(5))

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
            default = []

            print(vitals_data_df.head(5))

            #Time Variables
            start_time = []
            end_time = []
            count = 0
            PRNs = []

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
                    if tag == "Retro":
                        PRNs.append(row['SedPRN'])

                        if row['Default'] ==  'Y':
                            default.append('Y')
                        else:
                            default.append('N')
                    # Calculate the relative time within the window
                    in_window['dts'] = in_window['dts'] - row['start_time']

                    index = 0
                    for vital in vitals_list:
                        column = names[index]
                        temp_list = in_window[column].tolist()
                        vital.append(temp_list)
                        index+=1
                else: 
                    count+=1

            # Save Start/End times in correct format
            start_time_str = [ts.isoformat() for ts in start_time]
            end_time_str = [ts.isoformat() for ts in end_time]

            # Convert sbs to a numpy array
            sbs = np.array(sbs)
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
            default = np.array(default)
            
            # Further processing and saving...
            print('Save to file')

            # Remove empty lists from vitals_list and corresponding elements from names
            vitals_list_filtered = [v for v, n in zip(vitals_list, names) if v]
            names_filtered = [n for v, n in zip(vitals_list, names) if v]

            filename = f'{patient}_SICKBAY_{lead_time}MIN_{window_size-lead_time}MIN_{tag}.mat'
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
                    expected_samples = window_size * 30 # Time(min) * 60 sec/min * sr(1sample/2 sec)
                    if(len(cur_list[j]) > expected_samples):
                        cut = len(cur_list[j])-expected_samples
                        cur_list[j] = cur_list[j][cut:]

                    elif(len(cur_list[j]) < expected_samples): # Linear extrapolation to make all subarrays the same length
                        # Append NaN values to the end of the list
                        num_missing_samples = expected_samples - len(cur_list[j])
                        nan_values = np.full(num_missing_samples, np.nan)
                        cur_list[j] = np.concatenate((cur_list[j], nan_values))
                cur_list = np.array(cur_list, np.dtype('float16')) # Save List of np arrays as an np array
            
            filtered_dict['sbs'] = np.array(sbs)
            
            # Filter Vitals Data:
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
                # print(f'original {vitals_names_final[j]} vitals array shape: {np.array(temp_vitals[j]).shape} ')
                for i in range(len(vitals_SBS)):
                    if (filtering.checkVitals(temp_vitals[j][i], window_size, vitals_names_final[j])): # Check the data in a single window
                        vitals_list_final[j].append(temp_vitals[j][i]) # Append that single window data to the 2D hr, rr, spo2, bpm, bps, bpd arrays if that window's data is valid
                    else:
                        vitals_list_final[j].append(flag_list) # Append an array of zeros for window number i for the jth vitals metric if the data is invalid (i.e. too many NaN points)
                        # print(f'{vitals_names_final[j]} SBS index {i} has insufficient data, zeros appended in place') 
                # print(f'final {vitals_names_final[j]} vitals array shape: {np.array(vitals_list_final[j]).shape}') # The number of SBS scores by the number of samples in a window
            
            vitals_list_filtered_final = [v for v, n in zip(vitals_list_final, vitals_names_final) if v]
            names_filtered_final = [n for v, n in zip(vitals_list_final, vitals_names_final) if v]
            filtered_dict_final = {name: vitals for name, vitals in zip(names_filtered_final, vitals_list_filtered_final)}
            
            if tag == 'Retro':
                print('PRNs added')
                filtered_dict_final['SedPRN'] = PRNs

            filtered_dict_final['start_time'] = np.array(start_time_str, dtype=object)
            filtered_dict_final['end_time'] = np.array(end_time_str, dtype=object)
            filtered_dict_final['sbs'] = np.array(vitals_SBS)
            filtered_dict_final['default'] = np.array(default)

            savemat(save_file, filtered_dict_final, appendmat = False)
            print(f"{patient} has {count} SBS scores where vitals data does not line up in the time window")

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

def slice_data_by_time(data_dict, final_time):
    # Extract the start_times from the dictionary
    start_times = data_dict['start_time']
    
    # Find the indices where the start_time is less than final_time
    valid_indices = [i for i, start_time in enumerate(start_times) if start_time < final_time]
    
    # Slice all metrics including start_time based on the valid indices
    sliced_data = {
        key: [value[i] for i in valid_indices] if isinstance(value, list) else value[valid_indices]
        for key, value in data_dict.items()
    }
    
    return sliced_data



def load_and_segment_data_mat(data_dir, final_times_dict, window_size=15, lead_time=15, tag = ""):
    '''
    Load actigraphy and vitals waveform MAT file from a directory and segment it into time windows. 
    PatientX
    |_PatientX_AccelData.gt3x
    |_PatientX_SickBayData.mat
    '''
    # load_segment_sickbay(data_dir, window_size, lead_time, tag)
    # search for patient directories in the data directory
    for patient in os.listdir(data_dir):
        print(f'Patient: {patient}')
        final_time = final_times_dict[patient]

        # filter out non-directories
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)

            # SBS Scores from MAT File
            print('Loading SBS data')
            vitals_sbs_file = os.path.join(patient_dir, f'{patient}_SICKBAY_{lead_time}MIN_{window_size - lead_time}MIN_{tag}.mat') #requires load_segment_sickbay to be run before
            if not os.path.isfile(vitals_sbs_file):
                # raise FileNotFoundError(f'Actigraphy file not found: {sbs_file}')
                print("SBS File not found")
                continue 

            print('Loading actigraphy data')
            actigraphy_filepath = os.path.join(patient_dir, patient + '_AccelData.gt3x')
            if not os.path.isfile(actigraphy_filepath):
                raise FileNotFoundError(f'Actigraphy file not found: {actigraphy_filepath}')
            acti_data, acti_names = load_gt3x_data(actigraphy_filepath)
            acti_data['mag'] = np.linalg.norm(acti_data[['X', 'Y', 'Z']].values, axis=1)
            acti_data['dts'] = pd.to_datetime(acti_data['Timestamp'], unit='s')
            print(acti_data.shape)
            print(acti_names)
            
            # Implement error handling here if file does not exist...
            vitals_data = loadmat(vitals_sbs_file)
            # print(vitals_data)
            SBS = vitals_data['sbs'].flatten()
            default = vitals_data['default'].flatten()
            if tag == 'Retro':
                SedPRN = vitals_data['SedPRN'].flatten()
            # Flatten the nested arrays
            start_time_flat = vitals_data['start_time'].flatten()
            end_time_flat = vitals_data['end_time'].flatten()

            # Convert the flattened arrays to Timestamp objects
            start_time = [pd.Timestamp(str(ts[0])) for ts in start_time_flat]
            end_time = [pd.Timestamp(str(ts[0])) for ts in end_time_flat]

            if tag == 'Retro':
                epic_data = pd.DataFrame({
                    'SBS': SBS,
                    'start_time': start_time,
                    'end_time': end_time, 
                    'SedPRN' : SedPRN,
                    'Default' : default
                })
            else: 
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
            default_accel = []
            
            hr = vitals_data['hr']
            SpO2 = vitals_data['spo2']
            rr = vitals_data['rr']
            bps = vitals_data['bps']
            bpm = vitals_data['bpm']
            bpd = vitals_data['bpd']
            
            # Create new start_time/end_time variables that are for acti_data
            matched_start_times = []
            matched_end_times = []
            PRNs = []
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
                    if tag == "Retro":
                        PRNs.append(row["SedPRN"])
                    if row['Default'] == 'Y':
                        default_accel.append('Y')
                    else:
                        default_accel.append('N')
                        
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

            # matched_start_times_str = [ts.isoformat() for ts in matched_start_times]

            save_file = os.path.join(patient_dir, vitals_sbs_file)

            if tag == "Retro":
                data_dict = dict([('x_mag', x_mag), ('heart_rate', hr), 
                                     ('SpO2', SpO2), ('respiratory_rate', rr), ('blood_pressure_systolic', bps), 
                                     ('blood_pressure_mean', bpm), ('blood_pressure_diastolic', bpd), ('sbs', sbs), ('start_time', matched_start_times), ('PRNs', PRNs), ('Default', default_accel)])
            else: 
                data_dict = dict([('x_mag', x_mag), ('heart_rate', hr), 
                            ('SpO2', SpO2), ('respiratory_rate', rr), ('blood_pressure_systolic', bps), 
                            ('blood_pressure_mean', bpm), ('blood_pressure_diastolic', bpd), ('sbs', sbs), ('start_time', matched_start_times)])
                
            if final_time is not None:
                sliced_data = slice_data_by_time(data_dict, final_time) #Assumes final_time and start_time are TimeStamp Objects
            else: 
                sliced_data = data_dict
            
            sliced_data['start_time'] = [ts.isoformat() for ts in sliced_data['start_time']]

            savemat(save_file, sliced_data)

            print("Processing Complete")

def load_and_segment_data_excel(data_dir, final_times, window_size=10, lead_time=10, tag = ""):
    '''
    *** OPTIONAL CODE FOR ACCELEROMETRY ONLY ANALYSIS
    
    |_Load actigraphy and EPIC data from a directory and segment it into time windows.

    |_Assume that data_dir contains a directory for each patient, and all directories in data_dir are patient directories. Each patient directory must contain the actigraphy file and the EPIC file. 

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
        print(f'Patient: {patient}')
        final_time = final_times[patient]
        # filter out non-directories
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)
            
            # SBS Scores from Excel File
            print('Loading SBS data')
            s = f"_SBS_Scores_{tag}.xlsx"
            sbs_file = os.path.join(patient_dir, patient + s)
            if not os.path.isfile(sbs_file):
                # raise FileNotFoundError(f'Actigraphy file not found: {sbs_file}')
                print("SBS File not found")
                continue 


            print('Loading actigraphy data')
            actigraphy_filepath = os.path.join(patient_dir, patient + '_AccelData.gt3x')
            if not os.path.isfile(actigraphy_filepath):
                raise FileNotFoundError(f'Actigraphy file not found: {actigraphy_filepath}')
            acti_data, acti_names = load_gt3x_data(actigraphy_filepath)
            acti_data['mag'] = np.linalg.norm(acti_data[['X', 'Y', 'Z']].values, axis=1)
            acti_data['dts'] = pd.to_datetime(acti_data['Timestamp'], unit='s')
            print(acti_data.shape)
            print(acti_names)
        
            epic_data, epic_names = load_from_excel(sbs_file)
            epic_data.dropna(subset=['SBS'], inplace = True) # drop rows with missing SBS scores
            print(epic_data.shape)
            print(epic_names)
            epic_data['dts'] = pd.to_datetime(epic_data['Time_uniform'], format='%m/%d/%Y %H:%M:%S %p')
            # precompute start and end time for each SBS recording
            epic_data['start_time'] = epic_data['dts'] - pd.Timedelta(lead_time, 'minutes')
            epic_data['end_time'] = epic_data['dts'] + pd.Timedelta(window_size - lead_time, 'minutes')

            if(tag == "Retro"):
                epic_data = epic_data[(epic_data['Default'] != 'Y') & (epic_data['SBS'] != '')]
            
            print('Processing')
            windows = []
            sbs = []
            start_times = []
            PRNs = []
            for i, row in epic_data.iterrows():
                # don't like the for-loop, but its not a big bottleneck for the number of SBS recordings we are getting right now. 

                in_window = acti_data[(acti_data['dts'] > row['start_time']) & (acti_data['dts'] < row['end_time'])].loc[:, ['dts', 'mag']]
                in_window.rename(columns={'mag': f'mag_{i}'}, inplace=True)
                if in_window.shape[0] > 0:
                    sbs.append(row['SBS'])
                    start_times.append(row['start_time'])
                    in_window['dts'] = in_window['dts'] - row['start_time']
                    windows.append(in_window)

                    if tag == "Retro":
                        PRNs.append(row["SedPRN"])
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

            filename = f'{patient}_{lead_time}MIN_{window_size - lead_time}MIN_Accel_{tag}.mat'
            save_file = os.path.join(patient_dir, filename)

            del sliced_data
            gc.collect()

if __name__ == '__main__':
    '''
    Set the following:
    |_ data_dir: current working directory
    |_ window_size_in: total window used in analysis
    |_ lead_time_in: length of analysis before SBS score
    |_ tag: string tag of mat file
        E.g., _Validated, _Nurse, _WSTIM, etc.

    *** You must run load_segment_sickbay prior to load_and_segment_data_mat
    '''

    # data_dir = r'C:\Users\HP\Documents\JHU_Academics\Research\DT 6\NewPedAccel\VentilatedPatientData'
    data_dir = r'S:\Sedation_monitoring\PedAccel_directory\PedAccel\data_analysis\Vitals_accel_analysis\PatientData'

    # Define global variables
    heart_rate = []
    SpO2 = []
    respiratory_rate = []
    blood_pressure_systolic = []
    blood_pressure_mean = []
    blood_pressure_diastolic = []

    vitals_list = [heart_rate, SpO2, respiratory_rate, blood_pressure_systolic, blood_pressure_mean,blood_pressure_diastolic]
    names = ['heart_rate', 'SpO2', 'respiratory_rate', 'blood_pressure_systolic', 'blood_pressure_mean', 'blood_pressure_diastolic']

    window_size = 16
    lead_time = 15

    # tag = "Nurse"
    tag = "Retro"
    final_times_dict = {"Patient3": None, "Patient4": pd.Timestamp('2023-11-19 13:29:00'), "Patient9": None, "Patient11": pd.Timestamp('2024-02-01 18:00:00'), "Patient15": pd.Timestamp('2024-02-18 07:00:00')}  


    # load_segment_sickbay(data_dir, window_size, lead_time, tag)
    load_and_segment_data_mat(data_dir, final_times_dict, window_size, lead_time, tag)

    # add try catch block for naming

'''
    Data extraction Piepline Summary:

    Steps
    1) ccda_sbs_extraction. Runs on Safe desktop. Extracts SBS scores for a patient. 
        **How to incorporate retrospective scores: retro scores are formatted the same as ccda extracted scores and fit into the pipeline in the same way. 

    2) sickbay_vitals extraction. Runs on Safe desktop and extracts vitals data for a patient. 
    3) save the gt3x file from accelerometry

    4) load_segment_sickbay. This will filter the vitals data, replacing nan values and erreneous values. If too many invalid values are associated with a single SBS score and 
    vitals metric (i.e. more than 5 HR values were -10), then all measurements of that vitals metric associated with that SBS score will be replaced by an array of zero, 
    serving as a flag that this particular score is invalid for this particular vitals metric later in analysis. 
    5) Run load_and_segment_data_mat. This will produce a final .mat file with SBS, accelerometry, and Sickbay vitals. This file will be opened and used for analysis. 
    Final .mat file saved as either
    "{patient}_{lead_time}MIN_{window_size - lead_time}MIN_Validated.mat" or 
    "{patient}_{lead_time}MIN_{window_size - lead_time}MIN_Nurse.mat"

    6) Optionally run sickbay_mar extraction on safe desktop to collect mar data as well. 

    **gt3x files, SBS extraction, and sickbay extraction Data link: https://drive.google.com/drive/folders/1ZwHph6pqXW_QsIbGNAYLm9PNC-l9rVFg
    __________________________________________________________________________________________________________
    __________________________________________________________________________________________________________
    ECG Data Extraction Pipeline: 

    **ECG Data extraction is conucted in the ECG data folder. Regardless, the steps for ECG processing are below: 

    1) sickbay_ecg_extraction.py to create .mat files with ECG data and SBS scores around 15_1_window
    2) 
    3) 

    ECG Data link: https://livejohnshopkins-my.sharepoint.com/personal/sraghav9_jh_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsraghav9%5Fjh%5Fedu%2FDocuments%2FPICU%20Study%20Team%20De%2Didentified%20Data%2FPICU%5FECG%5FExtract&sortField=FileLeafRef&isAscending=true&e=5%3Af412fdd8d8c44a4b8365bc85b29451c1&sharingv2=true&fromShare=true&at=9&CT=1736100591204&OR=OWA%2DNT%2DMail&CID=57c7b448%2D8f68%2D3e1c%2D3917%2D16437e42f503&clickParams=eyJYLUFwcE5hbWUiOiJNaWNyb3NvZnQgT3V0bG9vayBXZWIgQXBwIiwiWC1BcHBWZXJzaW9uIjoiMjAyNDEyMTMwMDIuMTEiLCJPUyI6IldpbmRvd3MgMTEifQ%3D%3D&cidOR=Client&FolderCTID=0x012000334AC779193355408C0BCC6F3CAABE8A&view=0
'''    