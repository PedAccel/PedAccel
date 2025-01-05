from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import os

data_dir = os.path.join(os.path.dirname(__file__), 'data')
df = pd.read_csv(os.path.join(data_dir, 'pat2.csv'))

# original_sampling_rate = 250
# target_sampling_rate = 125

# nyquist_freq = target_sampling_rate / 2

# def butter_lowpass(cutoff, fs, order=5):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return b, a

# def apply_lowpass_filter(data, cutoff, fs, order=5):
#     b, a = butter_lowpass(cutoff, fs, order=order)
#     y = filtfilt(b, a, data)
#     return y

# cutoff_frequency = nyquist_freq
# ecg_1_filtered = apply_lowpass_filter(df['ECG_1'], cutoff_frequency, original_sampling_rate)
# ecg_2_filtered = apply_lowpass_filter(df['ECG_2'], cutoff_frequency, original_sampling_rate)
# ecg_3_filtered = apply_lowpass_filter(df['ECG_3'], cutoff_frequency, original_sampling_rate)

# resample_factor = target_sampling_rate / original_sampling_rate

# ecg_1_downsampled = resample(ecg_1_filtered, int(len(ecg_1_filtered) * resample_factor))
# ecg_2_downsampled = resample(ecg_2_filtered, int(len(ecg_2_filtered) * resample_factor))
# ecg_3_downsampled = resample(ecg_3_filtered, int(len(ecg_3_filtered) * resample_factor))

# time_downsampled = resample(df['Relative Time (sec)'], len(ecg_1_downsampled))

# df_downsampled = pd.DataFrame({
#     'Relative Time (sec)': time_downsampled[1000:len(time_downsampled)-1000],
#     'ECG_1': ecg_1_downsampled[1000:len(ecg_1_downsampled)-1000],
#     'ECG_2': ecg_2_downsampled[1000:len(ecg_2_downsampled)-1000],
#     'ECG_3': ecg_3_downsampled[1000:len(ecg_3_downsampled)-1000],})

# print("Data downsampled")

target_sampling_rate = 250
df_downsampled = df

df_cleaned = pd.DataFrame({
    'Relative Time (sec)': df_downsampled['Relative Time (sec)'],
    'ECG_1': nk.ecg_clean(df_downsampled['ECG_1'], sampling_rate=target_sampling_rate),
    'ECG_2': nk.ecg_clean(df_downsampled['ECG_2'], sampling_rate=target_sampling_rate),
    'ECG_3': nk.ecg_clean(df_downsampled['ECG_3'], sampling_rate=target_sampling_rate),})

print("Data cleaned")

signals1, info1 = nk.ecg_peaks(df_cleaned['ECG_1'], sampling_rate=target_sampling_rate, correct_artifacts=True)
signals2, info2 = nk.ecg_peaks(df_cleaned['ECG_2'], sampling_rate=target_sampling_rate, correct_artifacts=True)
signals3, info3 = nk.ecg_peaks(df_cleaned['ECG_3'], sampling_rate=target_sampling_rate, correct_artifacts=True)

print("Peaks found")

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(df_cleaned['Relative Time (sec)'], df_cleaned['ECG_1'], label='ECG_1')
plt.plot(df_cleaned['Relative Time (sec)'].iloc[info1['ECG_R_Peaks']], 
         df_cleaned['ECG_1'].iloc[info1['ECG_R_Peaks']], 'ro', label='R-peaks')
plt.legend()
plt.title('ECG_1 with R-peaks')

plt.subplot(3, 1, 2)
plt.plot(df_cleaned['Relative Time (sec)'], df_cleaned['ECG_2'], label='ECG_2')
plt.plot(df_cleaned['Relative Time (sec)'].iloc[info2['ECG_R_Peaks']], 
         df_cleaned['ECG_2'].iloc[info2['ECG_R_Peaks']], 'ro', label='R-peaks')
plt.legend()
plt.title('ECG_2 with R-peaks')

plt.subplot(3, 1, 3)
plt.plot(df_cleaned['Relative Time (sec)'], df_cleaned['ECG_3'], label='ECG_3')
plt.plot(df_cleaned['Relative Time (sec)'].iloc[info3['ECG_R_Peaks']], 
         df_cleaned['ECG_3'].iloc[info3['ECG_R_Peaks']], 'ro', label='R-peaks')
plt.legend()
plt.title('ECG_3 with R-peaks')

plt.tight_layout()
plt.show()

hrv_metrics = nk.hrv(signals1, sampling_rate=target_sampling_rate, show=True)
plt.show()