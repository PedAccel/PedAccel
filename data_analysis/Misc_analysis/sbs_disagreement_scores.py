'''
Nurse-Retrospective SBS Score Disagreement Quantification
'''

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

data_dir = r'\PatientData'

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

def find_closest_score(nurse_time, retro_times, max_diff=pd.Timedelta(minutes=5)):
    time_diff = abs(retro_times - nurse_time)
    closest_index = time_diff.idxmin()
    if time_diff[closest_index] <= max_diff:
        return closest_index
    return None

# Intraclass Correlation Coefficient (ICC)
def calculate_icc(x, y):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean_xy = np.mean([x, y])
    
    ssb = n * ((mean_x - mean_xy)**2 + (mean_y - mean_xy)**2)
    ssw = sum((x - mean_x)**2 + (y - mean_y)**2)
    
    msb = ssb / 1
    msw = ssw / (2*n - 1)
    
    icc = (msb - msw) / (msb + msw)
    return icc

def sbs_disagreement():
    for patient in os.listdir(data_dir):
        # filter out non-directories
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            # Load Nurse SBS Scores
            print('Loading Nurse SBS data')
            s = f"_SBS_Scores.xlsx"
            nurse_sbs_file = os.path.join(patient_dir, patient + s)
            if not os.path.isfile(nurse_sbs_file):
                print("SBS File not found")
                continue 

            nurse_epic_data, epic_names = load_from_excel(nurse_sbs_file)
            nurse_sbs = nurse_epic_data['SBS']
            nurse_sbs_time = pd.to_datetime(nurse_epic_data['Time_uniform'])

            # Load Retro Nurse SBS Scores
            print('Loading Retro SBS data')
            s = f"_SBS_Scores_Retro.xlsx"
            retro_sbs_file = os.path.join(patient_dir, patient + s)
            if not os.path.isfile(retro_sbs_file):
                print("SBS File not found")
                continue 

            retro_epic_data, epic_names = load_from_excel(retro_sbs_file)
            retro_sbs = retro_epic_data['SBS']
            retro_sbs_time = pd.to_datetime(retro_epic_data['Time_uniform'])

            # Calculate the difference between Nurse and Retro Nurse SBS scores
            matches = nurse_epic_data['Time_uniform'].apply(find_closest_score, args=(retro_epic_data['Time_uniform'],))

            matched_data = pd.DataFrame({
                'Nurse_Time': nurse_epic_data['Time_uniform'],
                'Nurse_SBS': nurse_epic_data['SBS'],
                'Retro_Time': nurse_epic_data['Time_uniform'][matches],
                'Retro_SBS': nurse_epic_data['SBS'][matches]
            })
            matched_data = matched_data.dropna()

            matched_data['Difference'] = abs(matched_data['Nurse_SBS'] - matched_data['Retro_SBS'])


            mean_diff = matched_data['Difference'].mean()

            # Percentage agreement
            exact_agreement = (matched_data['Difference'] == 0).mean() * 100

            # Cohen's Kappa
            kappa = cohen_kappa_score(matched_data['Nurse_SBS'], matched_data['Retro_SBS'])

            icc = calculate_icc(matched_data['Nurse_SBS'], matched_data['Retro_SBS'])

            plt.figure(figsize=(10, 6))
            plt.scatter(matched_data['Nurse_SBS'], matched_data['Retro_SBS'])
            plt.plot([0, 10], [0, 10], color='red', linestyle='--')  # Assuming SBS scores range from 0 to 10
            plt.xlabel('Nurse SBS Scores')
            plt.ylabel('Retrospective SBS Scores')
            plt.title('Comparison of Nurse and Retrospective SBS Scores')
            plt.show()
            print(f"Mean Absolute Difference: {mean_diff:.2f}")
            print(f"Exact Agreement: {exact_agreement:.2f}%")
            print(f"Cohen's Kappa: {kappa:.2f}")
            print(f"Intraclass Correlation Coefficient: {icc:.2f}")