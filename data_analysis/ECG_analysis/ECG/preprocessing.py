import os
import pandas as pd
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
from scipy.io import loadmat
from pygt3x.reader import FileReader 
from datetime import datetime

def load_mat(file_path):
    raw_data = loadmat(file_path)

    for key in raw_data.keys():
        if key != '__header__' and key != '__version__' and key != '__globals__':
            try:
                raw_data[key] = raw_data[key].squeeze()
            except:
                pass
    
    del raw_data['__header__']
    del raw_data['__version__']
    del raw_data['__globals__']

    return raw_data

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

def load_retro(data_dir, pat_num):
    retro_df = pd.read_csv(os.path.join(data_dir, f'Patient{pat_num}_SBS_Scores_Retro.csv'))
    retro_df = retro_df.dropna(axis=0, how='all')
    retro_df = retro_df.dropna(axis=1, how='all')

    retro_df.insert(0, 'Time', format_times(retro_df['Time_uniform']))
    retro_df.drop(columns=['Time_uniform', 'Datetime'], inplace=True)

    return retro_df

def load_sickbay(data_dir, pat_num):
    raw_sickbay = load_mat(os.path.join(data_dir, f'Patient{pat_num}_SICKBAY_10MIN_5MIN_Retro.mat'))

    for key in raw_sickbay.keys():
        if raw_sickbay[key].ndim == 2:
            new = np.empty(raw_sickbay[key].shape[0], dtype=object)

            for j in range(len(new)):
                new[j] = raw_sickbay[key]
            
            raw_sickbay[key] = new
    
    sickbay_df = pd.DataFrame(raw_sickbay)

    sickbay_df['start_time'] = format_times(sickbay_df['start_time'])
    sickbay_df['end_time'] = format_times(sickbay_df['end_time'])

    return sickbay_df

def load_ecg(data_dir, pat_num):
    raw_ecg = load_mat(os.path.join(data_dir, f'Patient{pat_num}_10MIN_5MIN_ECG_SBSFinal.mat'))

    ecg_df = pd.DataFrame(raw_ecg)

    ecg_df['start_time'] = format_times(ecg_df['start_time'])
    ecg_df['end_time'] = format_times(ecg_df['end_time'])

    return ecg_df

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
        sbs_score, start_time, end_time, ecg1, ecg2, ecg3 = load_data(file_path)
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
        sbs_score, start_time, end_time, _, _, _ = load_data(file_path)
        fs = 250

        quality_ecg1 = np.load(os.path.join(output_dir, pat_num+"quality_ecg1.npy"), allow_pickle=True)
        quality_ecg2 = np.load(os.path.join(output_dir, pat_num+"quality_ecg2.npy"), allow_pickle=True)
        quality_ecg3 = np.load(os.path.join(output_dir, pat_num+"quality_ecg3.npy"), allow_pickle=True)

    df_ecg1 = pd.DataFrame(columns=['start_time', 'end_time'])
    df_ecg2 = pd.DataFrame(columns=['start_time', 'end_time'])
    df_ecg3 = pd.DataFrame(columns=['start_time', 'end_time'])

    for i in tqdm(range(len(sbs_score))):
        if quality_ecg1[i].size > 4e3:
            metrics_1 = get_metrics(sbs_score[i], quality_ecg1[i], fs)
            metrics_1.insert(0, 'start_time', start_time[i])
            metrics_1.insert(1, 'end_time', end_time[i])
            df_ecg1 = pd.concat([df_ecg1, metrics_1], ignore_index=True)

        if quality_ecg2[i].size > 4e3:
            metrics_2 = get_metrics(sbs_score[i], quality_ecg2[i], fs)
            metrics_2.insert(0, 'start_time', start_time[i])
            metrics_2.insert(1, 'end_time', end_time[i])
            df_ecg2 = pd.concat([df_ecg2, metrics_2], ignore_index=True)
        
        if quality_ecg3[i].size > 4e3:
            metrics_3 = get_metrics(sbs_score[i], quality_ecg3[i], fs)
            metrics_3.insert(0, 'start_time', start_time[i])
            metrics_3.insert(1, 'end_time', end_time[i])
            df_ecg3 = pd.concat([df_ecg3, metrics_3], ignore_index=True)
    
    df_ecg1.to_csv(os.path.join(output_dir, pat_num+'df_ecg1.csv'), index=False)
    df_ecg2.to_csv(os.path.join(output_dir, pat_num+'df_ecg2.csv'), index=False)
    df_ecg3.to_csv(os.path.join(output_dir, pat_num+'df_ecg3.csv'), index=False)

    return df_ecg1, df_ecg2, df_ecg3

def load_dfs(output_dir, data_dir, i):
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

    return df_ecg1, df_ecg2, df_ecg3

def load_gt3x(file_path):
    with FileReader(file_path) as reader:
        df = reader.to_pandas()
        df.reset_index(inplace = True)

    return df

def format_times(times):
    t = np.empty(len(times), dtype='datetime64[ns]')
    
    for i, time in enumerate(times):
        if isinstance(time, (list, np.ndarray)):
            time = time[0]
        
        if isinstance(time, float):
            t[i] = np.datetime64(int(time), 'ns')
            continue

        if 'T' in time:
            try:
                t[i] = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S')
            except:
                raise ValueError(f'Unrecognized date format: {time}')
        else:
            try:
                t[i] = datetime.strptime(time, '%m/%d/%y %H:%M')
            except:
                raise ValueError(f'Unrecognized date format: {time}')

    return t

def match_times(df1, df2):
    """
    Concatenates two dataframes based on matching times

    Args:
        df1 (pd.DataFrame): Dataframe with 'Time'
        df2 (pd.DataFrame): Dataframe with 'start_time' and 'end_time'

    Returns:
        pd.DataFrame: Concatenated dataframe with matching times
    """
    # Convert 'start_time' and 'end_time' columns in df2 to datetime objects for valid comparison
    df2['start_time'] = pd.to_datetime(df2['start_time'])
    df2['end_time'] = pd.to_datetime(df2['end_time'])
    
    index1 = 0
    index2 = 0

    mask1 = np.zeros(len(df1), dtype=bool)
    mask2 = np.zeros(len(df2), dtype=bool)

    while index1 < len(df1) and index2 < len(df2):
        if df1['Time'][index1] < df2['start_time'][index2]:
            index1 += 1
        elif df1['Time'][index1] > df2['end_time'][index2]:
            index2 += 1
        else:
            mask1[index1] = True
            mask2[index2] = True

            index1 += 1
            index2 += 1

    df1 = df1[mask1].reset_index(drop=True)
    df2 = df2[mask2].reset_index(drop=True)

    return pd.concat([df1, df2], axis=1)

def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        print('Data directory does not exist')
        return

    # pat_nums = [4, 5, 6, 8, 9, 13]
    pat_nums = [4]

    for i in tqdm(pat_nums):
        df1, df2, df3 = load_dfs(output_dir, data_dir, i)

        print(df1.head())
        print(df2.head())
        print(df3.head())

if __name__ == "__main__":
    main()