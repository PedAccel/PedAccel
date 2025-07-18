'''
Extracts MAR Data from Sickbay .xlsx files
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

def load_sickbay_mar():

    patient_info_add = 'Full_Patient_List_CCDA.xlsx'
    patient_info_excel = load_workbook(patient_info_add)
    patient_info = patient_info_excel['Sheet2']

    print('Patient List Loaded')

    # sickbay_data_path = r"S:\Sedation_monitoring\CCDA_6771_Extract_03042024.xlsx"
    sickbay_data_path = r"S:\Sedation_monitoring\CCDA_6771_extract_IDENTIFIABLE_HIPPA\ccda6771_02192025.xlsx"
    sheet_name = 'Mar_Data'
    df = pd.read_excel(sickbay_data_path, sheet_name = sheet_name)

    print('SickBay Excel Loaded')

    for cell_a, cell_b, cell_c, cell_d in zip(patient_info['A'][1:], patient_info['B'][1:], patient_info['C'][1:], patient_info['D'][1:]):
        patient_num = cell_a.value
        patient_mrn = cell_b.value
        patient_newid = cell_c.value
        patient_enc_newid = cell_d.value

        print(f'Processing: Patient {patient_num}')

        criteria_id = df['pat_Newid'] == patient_newid

        filtered_rows = df[criteria_id]

        selected_columns = filtered_rows[['description', 'dose', 'mar_time', 'mar_action', 'med_name']]
        selected_columns.columns = ['description', 'dose', 'mar_time', 'mar_action', 'med_name']

        final_df = pd.DataFrame(selected_columns)

        data_dict = {col: final_df[col].values for col in final_df.columns}

        save_mat_dir = r"S:\Sedation_monitoring\Sickbay_extract\Extraction_II\Sickbay_mat_files"
        mat_file_path = os.path.join(save_mat_dir, f'Patient{patient_num}_SickBayMARData.mat')

        # Save the dictionary to a .mat file
        scipy.io.savemat(mat_file_path, data_dict)

        print(f"Dictionary saved to MATLAB .mat file: {mat_file_path}")
        print(f"Patinet {patient_num} Processing Complete")

if __name__ == '__main__':
    load_sickbay_mar()