'''
Extracts ECG data from SickBay CSV files and saves them as .mat files
|_This code is designed to run on the JHH SAFE Desktop Application
'''

# Import Modules
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from scipy.io import savemat
import os

directory = os.chdir(r'S:\Sedation_monitoring\Sickbay_extract\Extract_250Hz')

def load_from_excel(file_path):
    # Implement this function to load data from Excel
    # Return the data and column names
    data = pd.read_excel(file_path)
    return data, data.columns.tolist()

def load_sickbay_mar(window_size, lead_time):

    base_directory = r'S:\Sedation_monitoring\Sickbay_extract\Extract_250Hz'
    patient_info_add = 'Full_Patient_List_CCDA.xlsx'
    patient_info_excel = load_workbook(patient_info_add)
    patient_info = patient_info_excel['Sheet1']

    print('Patient List Loaded')

    for cell_a, cell_b in zip(patient_info['A'][1:], patient_info['B'][1:]):
        patient_num = cell_a.value
        patient_mrn = cell_b.value

        # Define dataframe and column names first...
        df_combined = pd.DataFrame(columns=['Relative Time (sec)', 'Time', 'GE_WAVE_ECG_1_ID', 'GE_WAVE_ECG_2_ID', 'GE_WAVE_ECG_3_ID'])
        
        patient_directory = os.path.join(base_directory, str(patient_mrn) + '_Study57_Tag123_EventList')
        print(patient_directory)

        sbs_file = os.path.join(patient_directory, f'Patient{patient_num}_SBS_Scores.xlsx')
        if not os.path.isfile(sbs_file):
            raise FileNotFoundError(f'SBS Scores not found: {sbs_file}')

        epic_data, epic_names = load_from_excel(sbs_file)
        epic_data.dropna(subset=['SBS'], inplace=True)
        epic_data['dts'] = pd.to_datetime(epic_data['Time_uniform'], format='mixed')
        epic_data['start_time'] = epic_data['dts'] - pd.Timedelta(lead_time, 'minutes')
        epic_data['end_time'] = epic_data['dts'] + pd.Timedelta(window_size - lead_time, 'minutes')
        print(len(epic_data))
        
        mat_file_path = os.path.join(patient_directory, f'Patient{patient_num}_ECG_SBSFinal.mat')
        initial_data = {
            'sbs_score': np.array([]),
            'start_time': np.array([], dtype='datetime64[ns]'),
            'end_time': np.array([], dtype='datetime64[ns]'),
            'ecg1': np.array([]),
            'ecg2': np.array([]),
            'ecg3': np.array([])
        }
        savemat(mat_file_path, initial_data)

        for index, row in epic_data.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            sbs_score = row['SBS']

            ecg_data = {'ecg1': [], 'ecg2': [], 'ecg3': []}

            for file_name in os.listdir(patient_directory):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(patient_directory, file_name)

                    # Error Check
                    if str(file_name) in file_path:
                        # Read the CSV file
                        chunk_iter = pd.read_csv(file_path, chunksize=10000000)
                        
                        chunk_counter = 0  # Initialize a counter for chunks
                        
                        for chunk in chunk_iter:

                            chunk['Time'] = pd.to_datetime(chunk['Time'])
                            chunk['Time_uniform'] = chunk['Time'].dt.strftime("%m/%d/%Y %I:%M:%S %p")
                            mask = (chunk['Time_uniform'] >= start_time) & (chunk['Time_uniform'] <= end_time)
                            if mask.any():
                                ecg_data['ecg1'].extend(chunk.loc[mask, 'GE_WAVE_ECG_1_ID'].tolist())
                                ecg_data['ecg2'].extend(chunk.loc[mask, 'GE_WAVE_ECG_2_ID'].tolist())
                                ecg_data['ecg3'].extend(chunk.loc[mask, 'GE_WAVE_ECG_3_ID'].tolist())

                            existing_data = savemat(mat_file_path)

                            existing_data['sbs_score'] = np.append(existing_data['sbs_score'], sbs_score)
                            existing_data['start_time'] = np.append(existing_data['start_time'], start_time.to_datetime64())
                            existing_data['end_time'] = np.append(existing_data['end_time'], end_time.to_datetime64())

                            for ecg_key in ['ecg1', 'ecg2', 'ecg3']:
                                ecg_segment = np.array(ecg_data[ecg_key]).reshape(1, -1)
                                if existing_data[ecg_key].size == 0:
                                    existing_data[ecg_key] = ecg_segment
                                else:
                                    existing_data[ecg_key] = np.vstack((existing_data[ecg_key], ecg_segment))

                            savemat(mat_file_path, existing_data)
                            del chunk
            print(f"Updated data for patient {patient_num}, SBS score {sbs_score}")

        print(f"Completed processing for patient MRN: {patient_mrn}")

if __name__ == '__main__':
    load_sickbay_mar(16, 15)