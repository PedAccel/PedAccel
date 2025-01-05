import os
import pandas as pd
from scipy.io import loadmat
import numpy as np

def save_chunk(file_path, file_name, start=None, end=None):
    raw_data = loadmat(file_path)

    if start is not None and end is not None:
        time = raw_data['Time'].squeeze()[start:end]
        relative_time = raw_data['Relative Time (sec)'].squeeze()[start:end]
        ecg_1 = raw_data['GE_WAVE_ECG_1_ID'].squeeze()[start:end]
        ecg_2 = raw_data['GE_WAVE_ECG_2_ID'].squeeze()[start:end]
        ecg_3 = raw_data['GE_WAVE_ECG_3_ID'].squeeze()[start:end]
        time_uniform = raw_data['Time_uniform'].squeeze()[start:end]
    else:
        time = raw_data['Time'].squeeze()
        relative_time = raw_data['Relative Time (sec)'].squeeze()
        ecg_1 = raw_data['GE_WAVE_ECG_1_ID'].squeeze()
        ecg_2 = raw_data['GE_WAVE_ECG_2_ID'].squeeze()
        ecg_3 = raw_data['GE_WAVE_ECG_3_ID'].squeeze()
        time_uniform = raw_data['Time_uniform'].squeeze()

    time_uniform_str = []
    for item in time_uniform:
        if isinstance(item, np.ndarray):
            date_str = item[0]
            time_uniform_str.append(date_str)
        else:
            date_str = str(item)
            time_uniform_str.append(date_str)

    datetime_format = '%m/%d/%Y %I:%M:%S %p'
    time_uniform_dt = pd.to_datetime(time_uniform_str, format=datetime_format)

    df = pd.DataFrame({
        'Time': time,
        'Relative Time (sec)': relative_time,
        'Time_uniform': time_uniform_dt,
        'ECG_1': ecg_1,
        'ECG_2': ecg_2,
        'ECG_3': ecg_3,
    })

    df.to_csv(file_name, index=False)

    print('Data saved' + file_name)

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    file_path = os.path.join(data_dir, 'Patient2_Event_Row_4_Data_zero_order_interpolation_chunk0.mat')

    save_chunk(file_path, os.path.join(output_dir, 'pat2.csv'))

if __name__ == '__main__':
    main()
