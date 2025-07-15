# %% [markdown]
# PRN Grading System

# %%
# Import Libraries & Set Parameters
import os 
from scipy.io import loadmat
import Actigraph_Metrics
import numpy as np
from scipy import stats
import math
import itertools 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import Actigraph_Metrics
from pygt3x.reader import FileReader

data_dir = '/Users/sidharthraghavan/Library/CloudStorage/OneDrive-Personal/Sid_stuff/PROJECTS/PedAccel/data_analysis/Vitals_accel_analysis/PatientData'
window_size = 100 # 100 is 1 second worth of time
lead_time = 1
slice_size_min = 31
Tag = "Retro"

min = 5

# %%
def load_from_excel(sbs_filepath, to_numpy=False, verbose=False):
    # Load data from Excel file
    df = pd.read_excel(sbs_filepath, header=0)
    col_names = df.columns.values.tolist()
    if 'SBS' not in col_names:
        raise ValueError('SBS column not found in the excel file')
    if to_numpy:
        array = df.to_numpy()
        return array, col_names
    return df, col_names

# %%
def extract_baseline(patient_dir, patient, TIME):
    from dateutil.parser import parse
    file_path = os.path.join(patient_dir, f'{patient}_SickBayData.mat')
    vitals_data = loadmat(file_path)
    vital_times_raw = vitals_data['time'].flatten()
    vital_times = []

    print('converting time data type')
    for t in vital_times_raw:
        if isinstance(t, np.ndarray):
            t_str = ''.join(chr(c) if isinstance(c, (int, np.integer)) else str(c) for c in t)
        else:
            t_str = str(t).strip()
        try:
            dt = datetime.strptime(t_str, '%m/%d/%Y %I:%M:%S %p')
            vital_times.append(dt)
        except ValueError as e:
            print(f"Could not parse time string: {t_str} — {e}")
    # Robustly convert TIME to datetime
    if not isinstance(TIME, datetime):
        try:
            TIME = parse(str(TIME))
        except Exception as e:
            print(f"Could not parse TIME: {TIME} — {e}")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    print('finished converting, extracting baseline data now')
    resp_rates = []
    heart_rates = []
    idx = 0
    for vitals_t in vital_times:
        time_diff = (TIME - vitals_t).total_seconds()
        if 0 < time_diff <= 21600:  # 6 hours
            resp_rates.append(vitals_data['respiratory_rate'].flatten()[idx])
            heart_rates.append(vitals_data['heart_rate'].flatten()[idx])
        idx+=1
    resp_rates = [x for x in resp_rates if np.isfinite(x)]
    heart_rates = [x for x in heart_rates if np.isfinite(x)]
    # Compute mean and IQR
    hr_mean = np.mean(heart_rates) if heart_rates else np.nan
    hr_q25 = np.percentile(heart_rates, 25) if heart_rates else np.nan
    hr_q75 = np.percentile(heart_rates, 75) if heart_rates else np.nan
    resp_mean = np.mean(resp_rates) if resp_rates else np.nan
    resp_q25 = np.percentile(resp_rates, 25) if resp_rates else np.nan
    resp_q75 = np.percentile(resp_rates, 75) if resp_rates else np.nan
    return hr_mean, hr_q25, hr_q75, resp_mean, resp_q25, resp_q75

# %%
from dateutil.parser import parse

#Place MAD data into SBS groups
window_size = 100 # 100 is 1 second worth of time
lead_time = 1
slice_size_min = 31
Tag = "Retro"

#Retrospective
for patient in os.listdir(data_dir):
    print(patient)
    if 'DS' in patient:
        continue
    # filter out non-directories
    patient_dir = os.path.join(data_dir, patient)
    if os.path.isdir(patient_dir):

        #Vitals Data
        vitals_file_path = os.path.join(patient_dir, f'{patient}_SICKBAY_{lead_time}MIN_{slice_size_min - lead_time}MIN_{Tag}.mat')
        if not os.path.isfile(vitals_file_path):
            continue
        vitals_data = loadmat(vitals_file_path)
        hr = vitals_data["heart_rate"]
        sbs = vitals_data['sbs'].flatten()
        x_mag = vitals_data["x_mag"]
        retro_PRN = vitals_data['PRNs'].flatten()
        resp = vitals_data["respiratory_rate"]
        times_sbs = vitals_data['start_time'].flatten()

        # Remove all white spaces from each string
        retro_PRN = [s.replace(" ", "") for s in retro_PRN]

        Times = []

        for i in range(len(sbs) - 1):
            if (retro_PRN[i] == 'Y') and (np.mean(resp[i]) != 0):
                TIME = times_sbs[i]
                # Robustly convert TIME to datetime
                if not isinstance(TIME, datetime):
                    try:
                        TIME = parse(str(TIME))
                    except Exception as e:
                        print(f"Could not parse TIME: {TIME} — {e}")
                        continue
                # Get baseline from 6 hours before this score
                hr_baseline, hr_q25, hr_q75, resp_baseline, resp_q25, resp_q75 = extract_baseline(patient_dir, patient, TIME)
                print(f'baseline is: {hr_baseline}')       
                print('Extracted Baseline')
                # Determine sampling rate and number of samples for last 15 minutes
                samples_per_min = int(60 * 0.5)  # 30 samples per minute at 0.5 Hz
                n_last = 15 * samples_per_min    # 450 samples for 15 min
                # x-axis in minutes (0 to 31)
                t_min = np.arange(len(hr[i])) / samples_per_min
                actual_values_hr = np.array(hr[i])
                actual_values_resp = np.array(resp[i])
                # Last 15 minutes (last 450 points)
                last_15_hr = actual_values_hr[-n_last:]
                last_15_rr = actual_values_resp[-n_last:]
                t_last_15 = t_min[-n_last:]
                avg_last_15_hr = np.mean(last_15_hr)
                avg_last_15_rr = np.mean(last_15_rr)
                # Check if in IQR
                hr_good = hr_q25 <= avg_last_15_hr <= hr_q75
                rr_good = resp_q25 <= avg_last_15_rr <= resp_q75
                prn_status = 'good' if hr_good and rr_good else 'bad'
                # Create PRN_Grading directory if it doesn't exist
                grading_dir = os.path.join(patient_dir, 'PRN_Grading')
                if not os.path.exists(grading_dir):
                    os.makedirs(grading_dir)
                # Plot HR with baseline and IQR
                plt.figure(figsize=(10, 4))
                plt.plot(t_min, actual_values_hr, label='Actual HR', color='red')
                plt.axhline(y=hr_baseline, color='blue', linestyle='--', label='HR Baseline (6h prior)')
                plt.fill_between(t_min, hr_q25, hr_q75, color='blue', alpha=0.2, label='HR Baseline IQR')
                plt.scatter(t_last_15, last_15_hr, color='black', s=10, label='Last 15-min HR')
                plt.axhline(y=avg_last_15_hr, color='black', linestyle=':', label=f'Last 15-min Avg HR ({avg_last_15_hr:.1f})')
                plt.title(f"HR Recovery for Event at {TIME} for {sbs[i]}\nPRN: {prn_status.upper()}")
                plt.xlabel("Time (minutes)")
                plt.xlim(0, 31)
                plt.ylabel("Heart Rate")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(grading_dir, f"HR_recovery_{i}.png"))
                plt.close()
                # Plot RESP with baseline and IQR
                plt.figure(figsize=(10, 4))
                plt.plot(t_min, actual_values_resp, label='Actual Resp', color='green')
                plt.axhline(y=resp_baseline, color='orange', linestyle='--', label='Resp Baseline (6h prior)')
                plt.fill_between(t_min, resp_q25, resp_q75, color='orange', alpha=0.2, label='Resp Baseline IQR')
                plt.scatter(t_last_15, last_15_rr, color='black', s=10, label='Last 15-min RR')
                plt.axhline(y=avg_last_15_rr, color='black', linestyle=':', label=f'Last 15-min Avg RR ({avg_last_15_rr:.1f})')
                plt.title(f"Respiratory Rate Recovery for Event at {TIME} for {sbs[i]}\nPRN: {prn_status.upper()}")
                plt.xlabel("Time (minutes)")
                plt.xlim(0, 31)
                plt.ylabel("Respiratory Rate")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(grading_dir, f"RR_recovery_{i}.png"))
                plt.close()
                # Compute errors
                hr_sse = np.sum((actual_values_hr - hr_baseline)**2)
                hr_signed_error = np.sum(actual_values_hr - hr_baseline)
                resp_sse = np.sum((actual_values_resp - resp_baseline)**2)
                resp_signed_error = np.sum(actual_values_resp - resp_baseline)
                print(f"\nErrors for event at {TIME}:")
                print(f"  HR SSE: {hr_sse:.2f}, HR Signed Error: {hr_signed_error:.2f}")
                print(f"  RESP SSE: {resp_sse:.2f}, RESP Signed Error: {resp_signed_error:.2f}")

                # ACCELEROMETRY (MAD) ANALYSIS
                # actual_values_mad = np.array(x_mag[i])
                # For baseline, use the 6-hour window from PatientX_AccelData.gt3x
                # accel_file_path = os.path.join(patient_dir, f'{patient}_AccelData.gt3x')
                # with FileReader(accel_file_path) as reader:
                #     accel_df = reader.to_pandas()
                # accel_df.reset_index(inplace=True)
                # # Compute vector magnitude
                # accel_df['mag'] = np.linalg.norm(accel_df[['X', 'Y', 'Z']].values, axis=1)
                # # Convert timestamp to datetime
                # accel_df['dts'] = pd.to_datetime(accel_df['Timestamp'], unit='s')
                # # Extract 6-hour window before TIME
                # window_end = TIME if isinstance(TIME, pd.Timestamp) else pd.to_datetime(TIME)
                # window_start = window_end - pd.Timedelta(hours=6)
                # mask = (accel_df['dts'] > window_start) & (accel_df['dts'] <= window_end)
                # mad_window = accel_df.loc[mask, 'mag'].values
                # # Compute MAD signal for the 6-hour window
                # if len(mad_window) > 0:
                #     mad_signal = Actigraph_Metrics.VecMag_MAD(mad_window, window=100)
                #     mad_mean = np.mean(mad_signal)
                #     mad_q25 = np.percentile(mad_signal, 25)
                #     mad_q75 = np.percentile(mad_signal, 75)
                # else:
                #     mad_mean = np.nan
                #     mad_q25 = np.nan
                #     mad_q75 = np.nan
                # Last 15 minutes (last 450 points)
                # last_15_mad = actual_values_mad[-n_last:]
                # t_last_15_mad = t_min[-n_last:]
                # avg_last_15_mad = np.mean(last_15_mad)
                # mad_good = mad_q25 <= avg_last_15_mad <= mad_q75
                # Plot MAD with baseline and IQR
                # plt.figure(figsize=(10, 4))
                # plt.plot(t_min, actual_values_mad, label='Actual MAD', color='purple')
                # plt.axhline(y=mad_mean, color='navy', linestyle='--', label='MAD Baseline (6h prior)')
                # plt.fill_between(t_min, mad_q25, mad_q75, color='navy', alpha=0.2, label='MAD Baseline IQR')
                # plt.scatter(t_last_15_mad, last_15_mad, color='black', s=10, label='Last 15-min MAD')
                # plt.axhline(y=avg_last_15_mad, color='black', linestyle=':', label=f'Last 15-min Avg MAD ({avg_last_15_mad:.2f})')
                # mad_status = 'good' if mad_good else 'bad'
                # plt.title(f"Accelerometry (MAD) Recovery for Event at {TIME} for {sbs[i]}\nPRN: {mad_status.upper()}")
                # plt.xlabel("Time (minutes)")
                # plt.xlim(0, 31)
                # plt.ylabel("Accelerometry (MAD)")
                # plt.legend()
                # plt.grid(True)
                # plt.savefig(os.path.join(grading_dir, f"MAD_recovery_{i}.png"))
                # plt.close()