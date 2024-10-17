# SickBay Data Extraction Code

import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl import Workbook
import scipy
# from scipy.io import savemat
import os
import re
os.chdir(r'S:\Sedation_monitoring')

def load_sickbay():

    patient_info_add = 'Full_Patient_List_CCDA.xlsx'
    patient_info_excel = load_workbook(patient_info_add)
    patient_info = patient_info_excel['Sheet1']

    print('Patient List Loaded')

    for cell_a, cell_b in zip(patient_info['A'][1:], patient_info['B'][1:]):
        patient_num = cell_a.value
        patient_mrn = cell_b.value

        print(f'Processing: Patient {patient_num}')

        filename = f'{patient_mrn}_SickBayData.csv'
        patient_dir = r"S:\Sedation_monitoring\Sickbay_extract\Extract_0.5Hz\Study57_Tag123_EventList"

        sickbay_data_path = os.path.join(patient_dir, filename)

        df = pd.read_csv(sickbay_data_path)

        print(df.head())

        # Convert time column to datetime
        df['Time'] = pd.to_datetime(df['Time'])
        df['Time_uniform'] = df['Time'].dt.strftime("%m/%d/%Y %I:%M:%S %p")
        
        # Convert DataFrame to a dictionary with keys as column names
        data_dict = {
            'time': df['Time_uniform'].values,
            'heart_rate': df['PARM_HR'].values,
            'SpO2': df['PARM_SPO2_M'].values,
            'respiratory_rate': df['PARM_RESP_RATE'].values,
            #'blood_pressure_systolic': [np.nan],
            'blood_pressure_systolic': df['PARM_NBP_SYS'].values,
            'blood_pressure_mean': df['PARM_NBP_MEAN'].values,
            # 'blood_pressure_diastolic': [np.nan]
            'blood_pressure_diastolic': df['PARM_NBP_DIA'].values
        }

        save_mat_dir = r"S:\Sedation_monitoring\Sickbay_extract\Sickbay_mat_files"
        mat_file_path = os.path.join(patient_dir, f'{patient_num}_SickBayData.mat')

        # Save the dictionary to a .mat file
        scipy.io.savemat(mat_file_path, data_dict)

        print(f"Dictionary saved to MATLAB .mat file: {mat_file_path}")
        print(f"Patinet {patient_num} Processing Complete")

if __name__ == '__main__':
    load_sickbay()