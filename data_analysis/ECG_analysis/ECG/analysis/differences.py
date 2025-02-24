import preprocessing
import os
import pandas as pd
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        print('Data directory does not exist')
        return

    pat_nums = [3, 4, 5, 6, 8, 9, 13]

    for i in tqdm(pat_nums):
        pat_num = 'pat' + str(i) + '_'
        file_path = os.path.join(data_dir, 'Patient' + str(i) + '_10MIN_5MIN_ECG_SBSFinal.mat')

        if not os.path.exists(output_dir):
            pass

        if not os.path.exists(os.path.join(output_dir, pat_num+'df_ecg1.csv')) or not os.path.exists(os.path.join(output_dir, pat_num+'df_ecg2.csv')) or not os.path.exists(os.path.join(output_dir, pat_num+'df_ecg3.csv')):
            df_ecg1, df_ecg2, df_ecg3 = preprocessing.save_metrics(file_path, pat_num, True)
        else:
            df_ecg1 = pd.read_csv(os.path.join(output_dir, pat_num+'df_ecg1.csv'))
            df_ecg2 = pd.read_csv(os.path.join(output_dir, pat_num+'df_ecg2.csv'))
            df_ecg3 = pd.read_csv(os.path.join(output_dir, pat_num+'df_ecg3.csv'))

        print('ECG 1 :')
        get_differences(df_ecg1, i, 1, True)
        print('ECG 2 :')
        get_differences(df_ecg2, i, 2, True)
        print('ECG 3 :')
        get_differences(df_ecg3, i, 3, True)

if __name__ == "__main__":
    main()