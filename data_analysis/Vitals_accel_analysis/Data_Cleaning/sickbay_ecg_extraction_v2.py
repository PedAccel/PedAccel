'''
Extracts ECG data from SickBay CSV files and saves them as .mat files
|_This code is designed to run on the JHH SAFE Desktop Application
|_Reads CSV files only once per patient
'''

# Import Modules
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from scipy.io import savemat, loadmat
import os

directory = os.chdir(r'S:\Sedation_monitoring\sickbay_extract\Extraction_I\Extract_250Hz')

def load_from_excel(file_path):
    # Implement this function to load data from Excel
    # Return the data and column names
    data = pd.read_excel(file_path)
    return data, data.columns.tolist()

def load_sickbay_mar(window_size, lead_time):

    base_directory = r'S:\Sedation_monitoring\sickbay_extract\Extraction_I\Extract_250Hz'
    patient_info_add = 'Full_Patient_List_CCDA.xlsx'
    patient_info_excel = load_workbook(patient_info_add)
    patient_info = patient_info_excel['Sheet1']

    print('Patient List Loaded')

    for cell_a, cell_b in zip(patient_info['A'][1:], patient_info['B'][1:]):
        patient_num = cell_a.value
        patient_mrn = cell_b.value
        
        patient_directory = os.path.join(base_directory, str(patient_mrn) + '_Study57_Tag123_EventList')
        print(f"Processing patient directory: {patient_directory}")

        sbs_file = os.path.join(patient_directory, f'Patient{patient_num}_SBS_Scores_Retro.xlsx')
        if not os.path.isfile(sbs_file):
            print(f'SBS Scores not found: {sbs_file}, skipping patient')
            continue

        epic_data, epic_names = load_from_excel(sbs_file)
        
        # For Retro Scores
        epic_data = epic_data[(epic_data['Default'] != 'Y') & (epic_data['SBS'] != '') & (epic_data['SBS'] != 'TODO')]
        epic_data.dropna(subset=['SBS'], inplace=True)
        epic_data['dts'] = pd.to_datetime(epic_data['Time_uniform'], format='mixed')
        epic_data['start_time'] = epic_data['dts'] - pd.Timedelta(lead_time, 'minutes')
        epic_data['end_time'] = epic_data['dts'] + pd.Timedelta(window_size - lead_time, 'minutes')
        
        print(f"Found {len(epic_data)} valid SBS scores for patient {patient_num}")
        
        if len(epic_data) == 0:
            print(f"No valid SBS scores for patient {patient_num}, skipping")
            continue
        
        # OPTIMIZATION: Read all CSV files once and store in memory
        print("Loading all ECG data for this patient...")
        all_ecg_data = pd.DataFrame()
        
        csv_files = [f for f in os.listdir(patient_directory) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found for patient {patient_num}, skipping")
            continue
            
        for file_name in csv_files:
            file_path = os.path.join(patient_directory, file_name)
            print(f"  Reading {file_name}...")
            
            try:
                # Read entire CSV at once (faster than chunking for this use case)
                chunk_data = pd.read_csv(file_path)
                chunk_data['Time_uniform'] = pd.to_datetime(chunk_data['Time'])
                
                # Only keep the columns we need
                chunk_data = chunk_data[['Time_uniform', 'GE_WAVE_ECG_2_ID']]
                
                # Append to master dataframe
                all_ecg_data = pd.concat([all_ecg_data, chunk_data], ignore_index=True)
                
            except Exception as e:
                print(f"  Error reading {file_name}: {e}")
                continue
        
        if all_ecg_data.empty:
            print(f"No ECG data loaded for patient {patient_num}, skipping")
            continue
            
        print(f"Loaded {len(all_ecg_data)} ECG data points")
        
        # Sort by time for faster searching
        all_ecg_data = all_ecg_data.sort_values('Time_uniform').reset_index(drop=True)
        
        # Initialize data structures for both window types
        mat_file_path = os.path.join(patient_directory, f'Patient{patient_num}_{lead_time}MIN_{window_size - lead_time}MIN_ECG_SBSRetro.mat')
        
        existing_data = {
            'sbs_score': [],
            'prn': [],
            'start_time': [],
            'end_time': [],
            'ecg2': []  # Only Lead II
        }
        
        # OPTIMIZATION: Process all SBS scores at once using vectorized operations
        print("Extracting ECG segments for all SBS scores...")
        
        for index, row in epic_data.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            sbs_score = row['SBS']
            prn = row['SedPRN']
            
            # Use boolean indexing to find data in time window (much faster)
            mask = (all_ecg_data['Time_uniform'] >= start_time) & (all_ecg_data['Time_uniform'] <= end_time)
            ecg_segment = all_ecg_data.loc[mask, 'GE_WAVE_ECG_2_ID'].values
            
            # Store the data
            existing_data['sbs_score'].append(sbs_score)
            existing_data['prn'].append(prn)
            existing_data['start_time'].append(start_time.to_datetime64())
            existing_data['end_time'].append(end_time.to_datetime64())
            existing_data['ecg2'].append(ecg_segment)
            
            print(f"  Extracted {len(ecg_segment)} ECG samples for SBS score {sbs_score}")
        
        # Convert lists to numpy arrays where appropriate
        existing_data['sbs_score'] = np.array(existing_data['sbs_score'])
        existing_data['prn'] = np.array(existing_data['prn'])
        existing_data['start_time'] = np.array(existing_data['start_time'])
        existing_data['end_time'] = np.array(existing_data['end_time'])
        # ecg2 stays as list of arrays due to variable lengths
        
        # Save to .mat file
        savemat(mat_file_path, existing_data)
        
        print(f"Completed processing for patient MRN: {patient_mrn}")
        print(f"Saved {len(existing_data['sbs_score'])} SBS episodes to {mat_file_path}")
        print("-" * 50)

if __name__ == '__main__':
    # Process both window types
    print("Processing window type 1: 16 minutes total, 15 minutes lead time...")
    load_sickbay_mar(35, 5)