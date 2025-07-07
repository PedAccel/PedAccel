'''
Extracts Patient Weight Data from CCDA .xlsx files
|_ To be run on JHH SAFE Desktop.

'''

# Import Modules
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl import Workbook
import scipy
import os
import re
os.chdir(r'S:\Sedation_monitoring')

def load_weight_data():

    # Define features for extraction
    desired_features = [
        'Weight'
    ]

    patient_info_add = 'Full_Patient_List_CCDA.xlsx'
    patient_info_excel = load_workbook(patient_info_add)
    patient_info = patient_info_excel['Sheet1']

    print('Patient List Loaded')

    sickbay_data_path = r"S:\Sedation_monitoring\ccda_extract_IDENTIFIABLE_HIPPA\CCDA_6771_Extract_03042024.xlsx"

    sheet_name = 'Flowsheet_Data'
    df = pd.read_excel(sickbay_data_path, sheet_name = sheet_name)

    print('SickBay Excel Loaded')

    for cell_a, cell_b in zip(patient_info['A'][1:], patient_info['B'][1:]):
        patient_num = cell_a.value
        patient_mrn = cell_b.value

        print(f'Processing: Patient {patient_num}')

        criteria_mrn = df['MRN'] == patient_mrn

        filtered_rows = df[criteria_mrn]

        feature_mask = filtered_rows['meas_disp_name'].isin(desired_features)
        selected_rows = filtered_rows[feature_mask]

        selected_columns = selected_rows[['Date_Time_start', 'Date_Time_end', 'meas_disp_name', 'question_response', 'display_name', 'assessment_datetime']]
        selected_columns.columns = ['Date_Time_start', 'Date_Time_end', 'meas_disp_name', 'question_response', 'display_name', 'assessment_datetime']

        print(selected_columns)
        output_file = f'Patient{patient_num}_weight_data.xlsx'
        selected_columns.to_excel(output_file, index=False)

        print(f'Saved weight data for Patient {patient_num}')
        
if __name__ == '__main__':
    load_weight_data()