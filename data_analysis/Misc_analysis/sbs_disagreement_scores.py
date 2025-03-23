'''
Nurse-Retrospective SBS Score Disagreement Quantification
'''

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
from sklearn.metrics import confusion_matrix


data_dir = r'C:\Users\sidha\OneDrive\Sid_stuff\PROJECTS\PedAccel\data_analysis\Misc_analysis\PatientData'

def load_from_excel(sbs_filepath, to_numpy=False, verbose=False):
    # Load data from Excel file
    df = pd.read_excel(sbs_filepath, header=0)
    col_names = df.columns.values.tolist()
    if 'SBS' not in col_names:
        raise ValueError('SBS column not found in the excel file')
    
    # Convert 'Time_uniform' to datetime
    if 'Time_uniform' in df.columns:
        df['Time_uniform'] = pd.to_datetime(df['Time_uniform'], errors='coerce')
    
    if to_numpy:
        array = df.to_numpy()
        return array, col_names
    return df, col_names

def sbs_disagreement():
    for patient in os.listdir(data_dir):
        # filter out non-directories
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print(f"\nProcessing patient: {patient}")
            # Load Nurse SBS Scores
            print('Loading Nurse SBS data')
            s = f"_SBS_Scores.xlsx"
            nurse_sbs_file = os.path.join(patient_dir, patient + s)
            if not os.path.isfile(nurse_sbs_file):
                print("SBS File not found")
                continue 

            nurse_epic_data, epic_names = load_from_excel(nurse_sbs_file)
            # nurse_epic_data = nurse_epic_data.dropna(subset=['Time_uniform'])
            nurse_sbs = nurse_epic_data['SBS']
            nurse_sbs_time = pd.to_datetime(nurse_epic_data['Time_uniform'], format='%Y-%m-%d %H:%M:%S', errors='coerce')


            # Load Retro Nurse SBS Scores
            print('Loading Retro SBS data')
            s = f"_SBS_Scores_Retro.xlsx"
            retro_sbs_file = os.path.join(patient_dir, patient + s)
            if not os.path.isfile(retro_sbs_file):
                print("SBS File not found")
                continue 

            retro_epic_data, epic_names = load_from_excel(retro_sbs_file)
            # retro_epic_data = retro_epic_data.dropna(subset=['Time_uniform'])
            retro_sbs = retro_epic_data['SBS']
            retro_sbs_time = pd.to_datetime(retro_epic_data['Time_uniform'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            
            print(f"Nurse data points: {len(nurse_sbs_time)}")
            print(f"Retro data points: {len(retro_sbs_time)}")

            # Find matching scores
            matched_scores = []
            score = 0
            for nurse_time, nurse_score in zip(nurse_sbs_time, nurse_epic_data['SBS']):
                for retro_time, retro_score in zip(retro_sbs_time, retro_epic_data['SBS']):
                    if pd.notna(nurse_time) and pd.notna(retro_time):
                        time_diff = abs(retro_time - nurse_time)
                        if time_diff <= pd.Timedelta(minutes=30):
                            if retro_score != 'TODO':
                                matched_scores.append((nurse_time, nurse_score, retro_score))
                                score = 1
                                break

            # Separate matched scores
            times, nurse_scores, retro_scores = zip(*matched_scores)

            # Convert scores to numeric values
            nurse_scores = pd.to_numeric(nurse_scores, errors='coerce')
            retro_scores = pd.to_numeric(retro_scores, errors='coerce')

            # Plot scores
            plt.figure(figsize=(12, 6))
            plt.plot(times, nurse_scores, 'b-', label='Nurse SBS')
            plt.plot(times, retro_scores, 'r-', label='Retrospective SBS')
            plt.xlabel('Time')
            plt.ylabel('SBS Score')
            plt.title(f'SBS Scores Comparison - {patient}')
            plt.legend()
            plt.savefig(os.path.join(patient_dir, f'{patient}_SBS_comparison.png'))
            plt.close()

            # Calculate disagreement scores
            mae = mean_absolute_error(nurse_scores, retro_scores)
            rmse = np.sqrt(mean_squared_error(nurse_scores, retro_scores))
            correlation = np.corrcoef(nurse_scores, retro_scores)[0, 1]
            kappa = cohen_kappa_score(nurse_scores, retro_scores)

            print(f"\nDisagreement metrics for patient {patient}:")
            print(f"Mean Absolute Error: {mae:.2f}")
            print(f"Root Mean Square Error: {rmse:.2f}")
            print(f"Correlation Coefficient: {correlation:.2f}")
            print(f"Cohen's Kappa: {kappa:.2f}")

            # Create confusion matrix
            cm = confusion_matrix(nurse_scores, retro_scores, labels=range(-3, 3))

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=range(-3, 3), yticklabels=range(-3, 3))
            plt.title(f'SBS Scores Confusion Matrix: {patient}')
            plt.xlabel('Retrospective SBS Score')
            plt.ylabel('Nurse SBS Score')
            plt.savefig(os.path.join(patient_dir, f'{patient}_SBS_confusion_matrix.png'))
            plt.close()

    return

sbs_disagreement()