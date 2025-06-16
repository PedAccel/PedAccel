import numpy as np
import pandas as pd
import neurokit2 as nk
import pywt
import scipy.signal as signal
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)

    df['Time'] = df['Time'] / 1e9

    fs = 1 / (df['Time'][1] - df['Time'][0])

    df_ecg = pd.DataFrame({
        'Time': df['Time'],
        'ECG_1': nk.ecg_clean(df['ECG_1'], sampling_rate=fs),
        'ECG_2': nk.ecg_clean(df['ECG_2'], sampling_rate=fs),
        'ECG_3': nk.ecg_clean(df['ECG_3'], sampling_rate=fs),
    })

    print(min(df_ecg['ECG_1']))
    print(max(df_ecg['ECG_1']))

    return fs, df_ecg

def main():
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'pat3.csv')

    fs, df_ecg = load_data(file_path)

    # Plot the ECG signals
    plt.figure(figsize=(10, 6))
    plt.plot(df_ecg['Time'], df_ecg['ECG_1'], label='ECG 1')
    plt.plot(df_ecg['Time'], df_ecg['ECG_2'], label='ECG 2')
    plt.plot(df_ecg['Time'], df_ecg['ECG_3'], label='ECG 3')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('ECG Signals')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()