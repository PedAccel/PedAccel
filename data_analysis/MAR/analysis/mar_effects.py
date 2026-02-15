import numpy as np
import pandas as pd
from scipy.signal import convolve

from mar_utils import filter_mar, classes, filter_drug
from mar_concentrations import calculate_doses

def get_drug_doses(mar_data):
    mar_narcotics, mar_paralytics, mar_alpha_agonists, mar_ketamines, mar_propofols, mar_etomidates, mar_benzodiazepines = filter_mar(mar_data)
    mar_classes = {'narcotics': mar_narcotics, 'paralytics': mar_paralytics, 'alpha_agonists': mar_alpha_agonists, 'ketamines': mar_ketamines, 'propofols': mar_propofols, 'etomidates': mar_etomidates, 'benzodiazepines': mar_benzodiazepines}

    drug_doses = {}

    for mar_class in mar_classes:
        if mar_classes[mar_class].shape[0] < 1:
            continue

        for drug_name in classes()[mar_class]:
            drug_df = filter_drug(mar_classes[mar_class], drug_name)
            
            if drug_df.shape[0] < 1:
                continue

            drug_doses[drug_name] = calculate_doses(drug_df)

    return drug_doses

def get_metrics(sickbay_data, accel_data):
    metrics = {
        'heart_rate': pd.DataFrame({
            'time': pd.to_datetime(sickbay_data['time']),
            'heart_rate': sickbay_data['heart_rate']}),
        'respiratory_rate': pd.DataFrame({
            'time': pd.to_datetime(sickbay_data['time']),
            'respiratory_rate': sickbay_data['respiratory_rate']}),
        'acceleration': pd.DataFrame({
            'time': pd.to_datetime(accel_data['time']),
            'acceleration': accel_data['a']}),}
    
    start = max(df['time'].min() for df in metrics.values())
    stop = min(df['time'].max() for df in metrics.values())
    
    triangle = triangle_filter(window=15)

    for metric, df in metrics.items():
        df = df[(df['time'] >= start) & (df['time'] <= stop)].reset_index(drop=True)
        df[metric] = df[metric].astype(float)

        df[metric] =  convolve(df[metric], triangle, mode="same", method="direct")
        df[metric] = (df[metric] - df[metric].mean()) / df[metric].std()

        metrics[metric] = df

    return metrics, start, stop

def get_drug_df(drug_doses, drug, start, stop):
    drug_df = drug_doses[drug]
    drug_df = drug_df[(drug_df['time'] >= pd.to_datetime(start)) & (drug_df['time'] <= pd.to_datetime(stop))].reset_index(drop=True)
    drug_df = drug_df[drug_df['bolus_dose'] != 0].reset_index(drop=True)

    return drug_df

def get_df(drug_df, metrics, start_before, end_before, start_after, end_after):
    times, doses = [], []

    hr_means_before = []
    hr_std_before = []
    hr_means_after = []
    hr_std_after = []

    rr_means_before = []
    rr_std_before = []
    rr_means_after = []
    rr_std_after = []

    a_means_before = []
    a_std_before = []
    a_means_after = []
    a_std_after = []

    for i in range(len(drug_df)):
        time = pd.to_datetime(drug_df.iloc[i, 0])
        dose = drug_df.iloc[i, 3]

        before_start = time - pd.Timedelta(minutes=start_before)
        before_end   = time + pd.Timedelta(minutes=end_before)

        after_start  = time - pd.Timedelta(minutes=start_after)
        after_end    = time + pd.Timedelta(minutes=end_after)

        hr = metrics['heart_rate']
        hr_before = hr[(hr['time'] >= before_start) & (hr['time'] <= before_end)].reset_index(drop=True)
        hr_after  = hr[(hr['time'] >= after_start)  & (hr['time'] <= after_end)].reset_index(drop=True)

        rr = metrics['respiratory_rate']
        rr_before = rr[(rr['time'] >= before_start) & (rr['time'] <= before_end)].reset_index(drop=True)
        rr_after  = rr[(rr['time'] >= after_start)  & (rr['time'] <= after_end)].reset_index(drop=True)

        a = metrics['acceleration']
        a_before = a[(a['time'] >= before_start) & (a['time'] <= before_end)].reset_index(drop=True)
        a_after  = a[(a['time'] >= after_start)  & (a['time'] <= after_end)].reset_index(drop=True)

        times.append(time)
        doses.append(dose)

        hr_means_before.append(np.mean(hr_before['heart_rate']))
        hr_std_before.append(np.std(hr_before['heart_rate']))
        hr_means_after.append(np.mean(hr_after['heart_rate']))
        hr_std_after.append(np.std(hr_after['heart_rate']))

        rr_means_before.append(np.mean(rr_before['respiratory_rate']))
        rr_std_before.append(np.std(rr_before['respiratory_rate']))
        rr_means_after.append(np.mean(rr_after['respiratory_rate']))
        rr_std_after.append(np.std(rr_after['respiratory_rate']))

        a_means_before.append(np.mean(a_before['acceleration']))
        a_std_before.append(np.std(a_before['acceleration']))
        a_means_after.append(np.mean(a_after['acceleration']))
        a_std_after.append(np.std(a_after['acceleration']))

    df = pd.DataFrame({
        'time': times,
        'dose': doses,
        'hr_mean_before': hr_means_before,
        'hr_std_before': hr_std_before,
        'hr_mean_after': hr_means_after,
        'hr_std_after': hr_std_after,
        'rr_mean_before': rr_means_before,
        'rr_std_before': rr_std_before,
        'rr_mean_after': rr_means_after,
        'rr_std_after': rr_std_after,
        'a_mean_before': a_means_before,
        'a_std_before': a_std_before,
        'a_mean_after': a_means_after,
        'a_std_after': a_std_after,})

    return df

def triangle_filter(window=15):
    L = window
    n = np.arange(L, dtype=float)
    center = (L - 1) / 2.0
    h = 1.0 - np.abs(n - center) / (center + 1.0)
    h = np.maximum(h, 0)

    h = h / np.sum(h)

    return h.astype(float)

