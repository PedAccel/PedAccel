import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'

for patient in os.listdir(data_dir):
    patient_dir = os.path.join(data_dir, patient)
    if os.path.isdir(patient_dir):
        print('Processing:', patient)
        print('Loading data')
            
        filename = f'{patient}_SICKBAY_10MIN_5MIN.mat'
            
        data_path = os.path.join(patient_dir, filename)
        data = loadmat(data_path)
        
        heart_rates = data['heart_rates'][0]  # Accessing the 1D array directly
        sbs = data['sbs'][0]  # Accessing the 1D array directly

        print("Shape of heart_rates:", heart_rates.shape)
        print("Shape of sbs:", sbs.shape)

        for sbs_value, heart_rate_val in zip(sbs, heart_rates):  # Iterate directly over the 1D arrays
            print(len(heart_rate_val[0]))
            t = np.arange(899) # Generate time array based on the length of heart_rate_val
            print(len(t))
            plt.figure()
            plt.title(f'{patient} Heart Rate vs Time for SBS Score {sbs_value}')
            plt.xlabel('Time (s)')
            plt.ylabel('Heart Rate (bpm)')
            plt.plot(t, heart_rate_val[0], color='red')
            plt.tight_layout()
            plt.ylim(0, 220)
            plt.xlim(0, 899)  # Set x-axis limits based on time array length
            plt.show()