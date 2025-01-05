import numpy as np
import pandas as pd
import neurokit2 as nk
import pywt
import scipy.signal as signal
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt

def load_data(fs):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    df = pd.read_csv(os.path.join(data_dir, 'pat2.csv'))

    # Convert 'Time_uniform' to datetime
    df['Time_uniform'] = pd.to_datetime(df['Time_uniform'])

    # Include 'Time_uniform' in the DataFrame
    df_ecg = pd.DataFrame({
        'Time_uniform': df['Time_uniform'],
        'ECG_1': nk.ecg_clean(df['ECG_1'], sampling_rate=fs),
        'ECG_2': nk.ecg_clean(df['ECG_2'], sampling_rate=fs),
        'ECG_3': nk.ecg_clean(df['ECG_3'], sampling_rate=fs),
    })
    
    return df_ecg

def detect_r_peaks(ecg_data, fs):
    _, rpeaks = nk.ecg_peaks(ecg_data, sampling_rate=fs, correct_artifacts=True)
    r_peaks_indices = rpeaks['ECG_R_Peaks']

    return r_peaks_indices

def calculate_rr_intervals(r_peaks, fs):
    rr_intervals = np.diff(r_peaks) / fs * 1000  # Convert to milliseconds
    
    return rr_intervals

def compute_ani(rr_intervals_ms, rr_interval_times):
    window_size = 64  # seconds
    sub_window_size = 16  # seconds
    alpha = 5.1
    beta = 1.2
    max_auc_total = 12.8  # Maximum possible AUC total

    # Convert rr_interval_times to seconds relative to the first time
    rr_time_sec = (rr_interval_times - rr_interval_times[0]) / np.timedelta64(1, 's')

    resample_fs = 8  # Hz
    sampling_interval = 1 / resample_fs  # in seconds
    rr_interp_func = interp1d(rr_time_sec, rr_intervals_ms, kind='linear', fill_value='extrapolate')
    time_axis = np.arange(0, rr_time_sec[-1], sampling_interval)
    rr_uniform_ms = rr_interp_func(time_axis)

    ani_values = []
    window_start_times = []  # To store the times corresponding to each ANI value
    window_samples = int(window_size * resample_fs)
    sub_window_samples = int(sub_window_size * resample_fs)

    # Ensure we have enough data for at least one window
    if len(rr_uniform_ms) < window_samples:
        print("Not enough data for one window. Exiting computation.")
        return ani_values, window_start_times

    for start in range(0, len(rr_uniform_ms) - window_samples + 1, sub_window_samples):
        window_rr_ms = rr_uniform_ms[start:start + window_samples]

        # Get the time corresponding to the start of the window
        window_start_time_sec = time_axis[start]
        window_start_time = rr_interval_times[0] + np.timedelta64(int(window_start_time_sec * 1e9), 'ns')
        window_start_times.append(window_start_time)

        # Normalization using Euclidean norm
        mean_rr = np.mean(window_rr_ms)
        centered_rr = window_rr_ms - mean_rr
        norm_value = np.sqrt(np.sum(centered_rr ** 2))
        if norm_value == 0:
            print("Norm value is zero. Skipping window.")
            continue  # Avoid division by zero
        norm_rr = centered_rr / norm_value

        # Wavelet transform to extract HF component
        coeffs = pywt.wavedec(norm_rr, 'db4', level=4)

        # Initialize all coefficients to zeros
        coeffs_HF = [np.zeros_like(c) for c in coeffs]

        # Include the detail coefficients that correspond to the HF band
        # Adjust the indices based on your desired HF band
        coeffs_HF[2] = coeffs[2]  # cD3
        coeffs_HF[3] = coeffs[3]  # cD2

        # Reconstruct the signal using selected coefficients
        hf_component = pywt.waverec(coeffs_HF, 'db4')

        # Truncate or pad hf_component to match the length of norm_rr
        hf_component = hf_component[:len(norm_rr)]

        # Shift hf_component to be >= 0
        hf_component -= np.min(hf_component)

        # Scale hf_component to range from 0 to 0.2 normalized units
        ptp = np.ptp(hf_component)
        if ptp == 0:
            print("HF component has zero peak-to-peak amplitude. Skipping window.")
            continue  # Avoid division by zero
        hf_component = (hf_component / ptp) * 0.2

        # Calculate AUCmin among sub-windows
        auc_mins = []
        for i in range(0, len(hf_component) - sub_window_samples + 1, sub_window_samples):
            sub_window = hf_component[i:i + sub_window_samples]

            # Find local maxima and minima
            peaks_max, _ = signal.find_peaks(sub_window)
            peaks_min, _ = signal.find_peaks(-sub_window)

            # If there are not enough peaks, skip this sub-window
            if len(peaks_max) < 2 or len(peaks_min) < 2:
                print(f"Not enough peaks in sub-window starting at index {i}. Skipping sub-window.")
                continue

            # Interpolate to get envelopes
            x = np.arange(len(sub_window))
            env_upper = np.interp(x, peaks_max, sub_window[peaks_max])
            env_lower = np.interp(x, peaks_min, sub_window[peaks_min])

            # Calculate area between envelopes
            auc = np.trapz(env_upper - env_lower, dx=sampling_interval)
            auc_mins.append(auc)

        if auc_mins:
            auc_min = min(auc_mins)
            # Ensure auc_min does not exceed maximum possible AUC total
            auc_min = min(auc_min, max_auc_total)

            # Calculate ANI
            ani = (100 * (alpha * auc_min + beta)) / max_auc_total
            ani = np.clip(ani, 0, 100)  # Ensure ANI is within 0-100

            ani_values.append(ani)
        else:
            print("No valid AUC values computed in this window.")

    return ani_values, window_start_times

def main():
    fs = 250  # Sampling frequency of the ECG data

    df = load_data(fs)
    ani_df = pd.DataFrame()

    for lead in df.columns:
        if lead == 'Time_uniform':
            continue
        print(f"Processing lead: {lead}")
        ecg_data = df[lead]

        r_peaks = detect_r_peaks(ecg_data, fs)

        if len(r_peaks) < 2:
            print(f"Not enough R-peaks detected in {lead}. Skipping.")
            continue

        rr_intervals = calculate_rr_intervals(r_peaks, fs)

        # Get the times corresponding to RR intervals
        r_peak_times = df['Time_uniform'].iloc[r_peaks].values.astype('datetime64[ns]')
        rr_interval_times = r_peak_times[1:]

        ani_values, window_start_times = compute_ani(rr_intervals, rr_interval_times)

        if ani_values:
            # Create a DataFrame for this lead
            lead_ani_df = pd.DataFrame({
                'Time': window_start_times,
                lead: ani_values
            })

            if ani_df.empty:
                ani_df = lead_ani_df
            else:
                # Merge on 'Time' column
                ani_df = pd.merge(ani_df, lead_ani_df, on='Time', how='outer')
        else:
            print(f"No ANI values computed for {lead}.")

    if not ani_df.empty:
        output_dir = os.path.join(os.path.dirname(__file__), 'data')
        ani_df.to_csv(os.path.join(output_dir, 'ani.csv'), index=False)
        print("ANI values saved to ani.csv.")

        # Plotting code adjusted to use the 'Time' column
        num_leads = len(ani_df.columns) - 1  # Exclude 'Time' column

        fig, axs = plt.subplots(num_leads, 1, figsize=(12, 4 * num_leads), sharex=True)

        if num_leads == 1:
            axs = [axs]  # Ensure axs is iterable

        for i, lead in enumerate([col for col in ani_df.columns if col != 'Time']):
            axs[i].plot(ani_df['Time'], ani_df[lead], label=lead)
            axs[i].set_ylabel('ANI Value')
            axs[i].set_title(f'ANI over Time for {lead}')
            axs[i].legend()
            axs[i].grid(True)

        axs[-1].set_xlabel('Time')
        plt.tight_layout()
        plt.show()
    else:
        print("No ANI values to save.")

if __name__ == "__main__":
    main()
