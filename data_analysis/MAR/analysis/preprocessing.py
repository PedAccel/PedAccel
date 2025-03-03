import os
from scipy.io import loadmat
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import math
from pygt3x.reader import FileReader 

def load_mar_data(data_dir, pat_num):
    """
    Loads MAR data from a .csv file and returns it as a pandas DataFrame.

    Parameters:
        data_dir (str): Path to the directory containing the .mat file.
        pat_num (int): Patient number. 

    Returns:
        pd.DataFrame: DataFrame containing MAR data.
    """    
    file_path = os.path.join(data_dir, f'Patient{pat_num}', f'Patient{pat_num}_SickBayMARData.csv')

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        raw_data = load_mat_file(os.path.join(data_dir, f'Patient{pat_num}', f'Patient{pat_num}_SickBayMARData.mat'))
        df = dict_to_df(raw_data, file_path)

    df = df.drop(columns=['description'])
    df.rename(columns={'mar_time': 'time'}, inplace=True)
    df = df[['time', 'dose', 'mar_action', 'med_name']]
    df['time'] = format_times(df['time'])
    df = df.sort_values(by='time', ascending=True).reset_index(drop=True)

    return df

def load_sickbay_data(data_dir, pat_num):
    """
    Loads SickBay data from a .mat file and returns it as a pandas DataFrame.

    Parameters:
        data_dir (str): Path to the directory containing the .mat file.
        pat_num (int): Patient number. 

    Returns:
        pd.DataFrame: DataFrame containing SickBay data.
    """
    file_path = os.path.join(data_dir, f'Patient{pat_num}', f'Patient{pat_num}_SickBayData.mat')

    raw_data = load_mat_file(file_path)

    df = pd.DataFrame(raw_data)

    df['time'] = format_times(df['time'])
    df = df.sort_values(by='time', ascending=True).reset_index(drop=True)

    return df

def load_sickbay_formatted_data(data_dir, pat_num):
    """
    Loads SickBay data from a .mat file and returns it as a pandas DataFrame.

    Parameters:
        data_dir (str): Path to the directory containing the .mat file.
        pat_num (int): Patient number. 

    Returns:
        pd.DataFrame: DataFrame containing SickBay data.
    """ 
    file_path = os.path.join(data_dir, f'Patient{pat_num}', f'Patient{pat_num}_SickBay_10MIN_5MIN_Retro.mat')

    raw_data= load_mat_file(file_path)

    for key in raw_data.keys():
        if raw_data[key].ndim == 2:
            new = np.empty(raw_data[key].shape[0], dtype=object)

            for j in range(len(new)):
                new[j] = raw_data[key]
            
            raw_data[key] = new
    
    sickbay_df = pd.DataFrame(raw_data)

    sickbay_df['start_time'] = format_times(sickbay_df['start_time'])
    sickbay_df['end_time'] = format_times(sickbay_df['end_time'])
    df = df.sort_values(by='start_time', ascending=True).reset_index(drop=True)

    return sickbay_df

def load_accel_data(data_dir, pat_num):
    """
    Loads accelerometer data from a .gt3x file and returns it as a pandas DataFrame.
    
    Parameters:
        data_dir (str): Path to the directory containing the .mat file.
        pat_num (int): Patient number.
        
    Returns:
        pd.DataFrame: DataFrame containing accelerometer data.
    """
    if os.path.exists(os.path.join(data_dir, f'Patient{pat_num}', f'Patient{pat_num}_AccelData.csv')):
        df = pd.read_csv(os.path.join(data_dir, f'Patient{pat_num}', f'Patient{pat_num}_AccelData.csv'))
        return df


    file_path = os.path.join(data_dir, f'Patient{pat_num}', f'Patient{pat_num}_AccelData.gt3x')

    df = load_gt3x_file(file_path)

    df = df[df["IdleSleepMode"] != True]
    df.rename(columns={'Timestamp': 'time'}, inplace=True)
    df['time'] = format_times(df['time'])
    df = df.sort_values(by='time', ascending=True).reset_index(drop=True)
    
    df = process_accel_data(df)
    df.to_csv(os.path.join(data_dir, f'Patient{pat_num}', f'Patient{pat_num}_AccelData.csv'), index=False)

    return df

def process_accel_data(df):
    """
    Finds the magnitude of the acceleration vector and averages it over 100 samples.

    Parameters:
        df (pd.DataFrame): DataFrame containing accelerometer data.

    Returns:
        pd.DataFrame: DataFrame containing processed accelerometer data.
    """
    df['a'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    new_df = pd.DataFrame(np.zeros((int(len(df)/100), 2)), columns=['time', 'a'])

    for i in range(len(new_df)):
        new_df.loc[i, 'time'] = df['time'].iloc[i*100]
        new_df.loc[i, 'a'] = np.average(df['a'].iloc[i*100:(i+1)*100])

    return new_df

def load_ecg_data(data_dir, pat_num):
    """
    Loads ECG data from a .csv file and returns it as a pandas DataFrame.
    
    Parameters:
        data_dir (str): Path to the directory containing the .mat file.
        pat_num (int): Patient number.
        
    Returns:
        pd.DataFrame: DataFrame containing ECG data.
    """
    file_path = os.path.join(data_dir, f'Patient{pat_num}', f'Patient{pat_num}_10MIN_5MIN_ECG_SBSFinal.mat')

    raw_data = load_mat_file(file_path)

    df = pd.DataFrame(raw_data)

    df['start_time'] = format_times(df['start_time'])
    df['end_time'] = format_times(df['end_time'])
    df = df.sort_values(by='start_time', ascending=True).reset_index(drop=True)

    return df

def load_retro_data(data_dir, pat_num):
    df = pd.read_csv(os.path.join(data_dir, 'Patient{pat_num}', f'Patient{pat_num}_SBS_Scores_Retro.csv'))
    
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')
    df.insert(0, 'time', format_times(df['Time_uniform']))
    df.drop(columns=['Time_uniform', 'Datetime'], inplace=True)
    df = df.sort_values(by='time', ascending=True).reset_index(drop=True)

    return df

def load_mat_file(file_path):
    """
    Loads .mat file and returns a dictionary.

    Parameters:
        file_path (str): Path to .mat file.

    Raises:
        FileNotFoundError: If the file does not exist.    

    Returns:
        dict: Dictionary containing data from .mat file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    raw_data = loadmat(file_path)

    for key in list(raw_data.keys()):
        if key.startswith("__"):
            del raw_data[key]
        else:
            try:
                raw_data[key] = raw_data[key].squeeze()
            except:
                pass

    return raw_data

def load_gt3x_file(file_path):
    """
    Loads .gt3x file and returns a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to .gt3x file.

    Raises:
        FileNotFoundError: If the file does not exist.
        
    Returns:
        pd.DataFrame: DataFrame containing data from .gt3x file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with FileReader(file_path) as reader:
        df = reader.to_pandas()
        df.reset_index(inplace = True)

    return df

def dict_to_df(raw_data, save_path=None):
    """
    Converts dictionary to pandas DataFrame and (optionally) saves it as a .csv file.

    Parameters:
        raw_data (dict): Dictionary containing data.
        save_path (str, optional): Path to save .csv file. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing data.
    """
    for key in raw_data.keys():
        try:
            raw_data[key] = raw_data[key].squeeze()
        except:
            pass

    df = pd.DataFrame(raw_data)

    if save_path is not None:
        df.to_csv(save_path, index=False)
    
    return df

def format_times(times):
    """
    Formats times to datetime64[ns] format.

    Parameters:
        times (list): List of times to format.

    Returns:
        np.ndarray: Array of formatted times.
    """
    t = np.empty(len(times), dtype='datetime64[ns]')
    
    for i, time in enumerate(times):
        if isinstance(time, (list, np.ndarray)):
            time = time[0]
        
        if isinstance(time, float):
            if int(math.log10(time)) == 9:
                t[i] = np.datetime64(int(time), 's')
            elif int(math.log10(time)) == 12:
                t[i] = np.datetime64(int(time), 'ms')
            elif int(math.log10(time)) == 15:
                t[i] = np.datetime64(int(time), 'us')
            elif int(math.log10(time)) == 18:
                t[i] = np.datetime64(int(time), 'ns')
            continue

        if 'T' in time:
            try:
                t[i] = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S')
            except:
                raise ValueError(f'Unrecognized date format: {time}')
        elif 'M' in time:
            try:
                t[i] = datetime.strptime(time, '%m/%d/%Y %I:%M:%S %p')
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
    Matches times between two DataFrames.
    
    Parameters:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing matched times.
    """
    if df1.empty or df2.empty:
        return pd.DataFrame()
    
    if 'time' in df1.columns and 'time' in df2.columns:
        return match_times_time_time(df1, df2)
    elif 'time' in df1.columns and 'start_time' in df2.columns and 'end_time' in df2.columns:
        return match_times_time_startend(df1, df2)
    elif 'start_time' in df1.columns and 'end_time' in df1.columns and 'time' in df2.columns:
        return match_times_time_startend(df2, df1)
    elif 'start_time' in df1.columns and 'end_time' in df1.columns and 'start_time' in df2.columns and 'end_time' in df2.columns:
        return match_times_startend_startend(df1, df2)
    else:
        print("Incompatible DataFrames")
        return pd.DataFrame()

def match_times_time_time(df1, df2):
    pass

def match_times_time_startend(df1, df2):
    """
    Concatenates two dataframes based on matching times

    Args:
        df1 (pd.DataFrame): Dataframe with 'time'
        df2 (pd.DataFrame): Dataframe with 'start_time' and 'end_time'

    Returns:
        pd.DataFrame: Concatenated dataframe with matching times
    """
    df2['start_time'] = pd.to_datetime(df2['start_time'])
    df2['end_time'] = pd.to_datetime(df2['end_time'])
    
    index1 = 0
    index2 = 0

    mask1 = np.zeros(len(df1), dtype=bool)
    mask2 = np.zeros(len(df2), dtype=bool)

    while index1 < len(df1) and index2 < len(df2):
        if df1['time'][index1] < df2['start_time'][index2]:
            index1 += 1
        elif df1['time'][index1] > df2['end_time'][index2]:
            index2 += 1
        else:
            mask1[index1] = True
            mask2[index2] = True

            index1 += 1
            index2 += 1

    df1 = df1[mask1].reset_index(drop=True)
    df2 = df2[mask2].reset_index(drop=True)

    return pd.concat([df1, df2], axis=1)

def match_times_startend_startend(df1, df2):
    pass

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    pat_nums = [9]

    for pat_num in tqdm(pat_nums):
        mar = load_mar_data(data_dir, pat_num)
        sickbay = load_sickbay_data(data_dir, pat_num)
        accel = load_accel_data(data_dir, pat_num)

        print(mar.keys())
        print(mar.shape)
        print(mar.head())
        print(mar.tail())

        print(sickbay.keys())
        print(sickbay.shape)
        print(sickbay.head())
        print(sickbay.tail())

        print(accel.keys())
        print(accel.shape)
        print(accel.head())
        print(accel.tail())

if __name__ == "__main__":
    main()