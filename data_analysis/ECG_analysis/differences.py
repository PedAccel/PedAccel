import os
import pandas as pd
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    raw_data = loadmat(file_path)

    sbs_score = raw_data['sbs_score'].squeeze()
    start_time = raw_data['start_time'].squeeze()
    end_time = raw_data['end_time'].squeeze()
    ecg1 = raw_data['ecg1'].squeeze()
    ecg2 = raw_data['ecg2'].squeeze()
    ecg3 = raw_data['ecg3'].squeeze()

    for i in range(len(sbs_score)):
        start_time[i] = start_time[i].squeeze()
        end_time[i] = end_time[i].squeeze()

    for i in range(len(ecg1)):
        ecg1[i] = ecg1[i].squeeze()
        ecg2[i] = ecg2[i].squeeze()
        ecg3[i] = ecg3[i].squeeze()

    return sbs_score, start_time, end_time, ecg1, ecg2, ecg3

def extract_quality_signal(signal, sample_rate):
    quality_dict = {}
    step = int(sample_rate)
    window_size = int(10*sample_rate)

    for i in range(0, len(signal)-window_size, step):
        quality = nk.ecg_quality(signal[i:i+window_size],rpeaks = None, sampling_rate = sample_rate, method = 'zhao2018', approach = None)
        if(quality == 'Excellent' or quality == 'Barely acceptable'):
            quality_dict[i] = 1
        else:
            quality_dict[i] = 0
    
    quality_vals = list(quality_dict.values())
    index_tuple = get_continuous_indices(quality_vals)
    quality_keys = list(quality_dict.keys())

    if quality_keys == [] or index_tuple == (-1, -1):
        return 0, 0

    return quality_keys[index_tuple[0]], quality_keys[index_tuple[1]]

def get_continuous_indices(arr):
    max_len = 0
    max_start = -1
    max_end = -1
    
    current_start = -1
    current_len = 0
    
    for i, value in enumerate(arr):
        if value == 1:
            if current_start == -1:
                current_start = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
                max_end = i - 1
            current_start = -1
            current_len = 0
    
    if current_len > max_len:
        max_len = current_len
        max_start = current_start
        max_end = len(arr) - 1
    
    return (max_start, max_end) if max_len > 0 else (-1, -1)

def get_metrics(sbs_score, ecg, fs):
    signals, _ = nk.ecg_peaks(ecg, sampling_rate=fs, correct_artifacts=True)
    df = nk.hrv(signals, sampling_rate=fs)
    df.insert(0, 'SBS_SCORE', sbs_score)

    return df

def save_metrics(file_path, pat_num, save=False):
    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    if not os.path.exists(os.path.join(output_dir, pat_num+"quality_ecg1.npy")) or not os.path.exists(os.path.join(output_dir, pat_num+"quality_ecg2.npy")) or not os.path.exists(os.path.join(output_dir, pat_num+"quality_ecg3.npy")):
        sbs_score, _, _, ecg1, ecg2, ecg3 = load_data(file_path)
        fs = 250
        # fs = int(1e9 * len(ecg1[0]) / (end_time[0] - start_time[0]))

        quality_ecg1 = np.empty(len(ecg1), dtype=object)
        quality_ecg2 = np.empty(len(ecg2), dtype=object)
        quality_ecg3 = np.empty(len(ecg3), dtype=object)

        for i in tqdm(range(len(sbs_score))):
            quality_start, quality_end = extract_quality_signal(ecg1[i], fs)
            quality_ecg1[i] = ecg1[i][quality_start:quality_end]

            quality_start, quality_end = extract_quality_signal(ecg2[i], fs)
            quality_ecg2[i] = ecg2[i][quality_start:quality_end]

            quality_start, quality_end = extract_quality_signal(ecg3[i], fs)
            quality_ecg3[i] = ecg3[i][quality_start:quality_end]

        if save:
            np.save(os.path.join(output_dir, pat_num+"quality_ecg1.npy"), quality_ecg1)
            np.save(os.path.join(output_dir, pat_num+"quality_ecg2.npy"), quality_ecg2)
            np.save(os.path.join(output_dir, pat_num+"quality_ecg3.npy"), quality_ecg3)
    else:
        sbs_score, _, _, _, _, _ = load_data(file_path)
        fs = 250

        quality_ecg1 = np.load(os.path.join(output_dir, pat_num+"quality_ecg1.npy"), allow_pickle=True)
        quality_ecg2 = np.load(os.path.join(output_dir, pat_num+"quality_ecg2.npy"), allow_pickle=True)
        quality_ecg3 = np.load(os.path.join(output_dir, pat_num+"quality_ecg3.npy"), allow_pickle=True)

    df_ecg1 = pd.DataFrame()
    df_ecg2 = pd.DataFrame()
    df_ecg3 = pd.DataFrame()

    for i in tqdm(range(len(sbs_score))):
        if quality_ecg1[i].size > 4e3:
            metrics_1 = get_metrics(sbs_score[i], quality_ecg1[i], fs)
            df_ecg1 = pd.concat([df_ecg1, metrics_1], ignore_index=True)

        if quality_ecg2[i].size > 4e3:
            metrics_2 = get_metrics(sbs_score[i], quality_ecg2[i], fs)
            df_ecg2 = pd.concat([df_ecg2, metrics_2], ignore_index=True)
        
        if quality_ecg3[i].size > 4e3:
            metrics_3 = get_metrics(sbs_score[i], quality_ecg3[i], fs)
            df_ecg3 = pd.concat([df_ecg3, metrics_3], ignore_index=True)
    
    df_ecg1.to_csv(os.path.join(output_dir, pat_num+'df_ecg1.csv'), index=False)
    df_ecg2.to_csv(os.path.join(output_dir, pat_num+'df_ecg2.csv'), index=False)
    df_ecg3.to_csv(os.path.join(output_dir, pat_num+'df_ecg3.csv'), index=False)

    return df_ecg1, df_ecg2, df_ecg3

def standardize_data(df_ecg):
    df_ecg = df_ecg.dropna(axis='columns')

    for key in df_ecg.keys():
        df_ecg[key] = StandardScaler().fit_transform(df_ecg[key].values.reshape(-1, 1))
    
    return df_ecg

def get_differences(df_ecg, pat_num=None, ecg_num=None, save=False):
    sbs_score = df_ecg['SBS_SCORE']
    df_ecg = df_ecg.drop(columns=['SBS_SCORE'])

    df_ecg = standardize_data(df_ecg)

    df_ecg.insert(0, 'SBS_SCORE', sbs_score)

    df_diff = pd.DataFrame()

    for i in range(1, len(df_ecg)):
        diff = df_ecg.iloc[i] - df_ecg.iloc[i - 1]
        sbs_prev = df_ecg['SBS_SCORE'].iloc[i - 1]
        sbs_curr = df_ecg['SBS_SCORE'].iloc[i]
        diff = pd.concat([pd.Series([sbs_prev, sbs_curr], index=['SBS_SCORE_-1', 'SBS_SCORE_0']), diff], axis=0)
        df_diff = pd.concat([df_diff, diff.to_frame().T], ignore_index=True)
    
    df_diff.rename(columns={'SBS_SCORE': 'DELTA_SBS_SCORE'}, inplace=True)
    
    if save:
        df_diff.to_csv(os.path.join(os.path.dirname(__file__), 'output', f'pat{pat_num}_ecg{ecg_num}_differences.csv'), index=False)
    
    return df_diff

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
            df_ecg1, df_ecg2, df_ecg3 = save_metrics(file_path, pat_num, True)
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