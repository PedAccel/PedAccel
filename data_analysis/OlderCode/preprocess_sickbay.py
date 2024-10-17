import pandas as pd
import numpy as np
import os
from scipy.io import savemat
from functools import reduce
from scipy.io import loadmat
import math 

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

def load_segment_sickbay(data_dir, window_size=15, lead_time=10):
    # Iterate through patient directories
    for patient in os.listdir(data_dir):
        for i in vitals_list:
            i.clear()  # Clears each list in-place

        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)

            # Load SBS data
            sbs_file = os.path.join(patient_dir, f'{patient}_SBS_Scores_Validated.xlsx')
            if not os.path.isfile(sbs_file):
                raise FileNotFoundError(f'EPIC file not found: {sbs_file}')
            epic_data, epic_names = load_from_excel(sbs_file)
            
            # Statement to exclude SBS scores without stimulation
            # epic_data = epic_data[epic_data['Stim?'] == 'Y']
            
            # Statement for Default SBS Score Processing (Score 4)
            for i in range(len(epic_data['SBS'])):
                if epic_data['Default?'][i] == 'Y':
                    epic_data['SBS'][i] = 4
            
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
            # ed
            start_time_str = [ts.isoformat() for ts in start_time]
            end_time_str = [ts.isoformat() for ts in end_time]

            # Convert sbs to a numpy array
            sbs = np.array(sbs)
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
            # print(start_time)
            
            # Further processing and saving...
            print('Save to file')

            # Remove empty lists from vitals_list and corresponding elements from names
            vitals_list_filtered = [v for v, n in zip(vitals_list, names) if v]
            names_filtered = [n for v, n in zip(vitals_list, names) if v]

            filename = f'{patient}_SICKBAY_{lead_time}MIN_{window_size-lead_time}MIN_Validated_Default.mat'
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

                    elif(len(cur_list[j]) < expected_samples): #linear extrapolation to make all subarrays the same length
                        # Append NaN values to the end of the list
                        num_missing_samples = expected_samples - len(cur_list[j])
                        nan_values = np.full(num_missing_samples, np.nan)
                        cur_list[j] = np.concatenate((cur_list[j], nan_values))
                        print(f'after sampling: {len(cur_list[j])}')
                cur_list = np.array(cur_list, np.dtype('float16')) #save List of np arrays as an np array

            filtered_dict['sbs'] = np.array(sbs)
            filtered_dict['start_time'] = np.array(start_time_str, dtype=object)
            filtered_dict['end_time'] = np.array(end_time_str, dtype=object)
            print(filtered_dict['start_time'])
            savemat(save_file, filtered_dict, appendmat = False)

if __name__ == '__main__':
    # data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
    data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'
    load_segment_sickbay(data_dir)