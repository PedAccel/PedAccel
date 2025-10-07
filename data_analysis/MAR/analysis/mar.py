import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def classes():
    """
    Define the classes of medications used in the analysis.

    Returns:
        dict: A dictionary containing the medication classes and their respective medications.
    """
    narcotics = ['fentanyl', 'morphine', 'hydromorphone', 'oxycodone', 'methadone', 'remifentanil']
    paralytics = ['rocuronium', 'vecuronium', 'succinylcholine', 'cisatracurium']
    alpha_agonists = ['dexmedetomidine', 'clonidine']
    ketamines = ['ketamine']
    propofols = ['propofol']
    etomidates = ['etomidate']
    benzodiazepines = ['midazolam', 'diazepam', 'lorazepam']

    classes = {'narcotics': narcotics, 
               'paralytics': paralytics, 
               'alpha_agonists': alpha_agonists, 
               'ketamines': ketamines, 
               'propofols': propofols, 
               'etomidates': etomidates, 
               'benzodiazepines': benzodiazepines}

    return classes

def half_lives():
    """
    Define the half-lives of medications used in the analysis.
    
    Returns:
        dict: A dictionary containing the half-lives of each medication in hours.
    """
    half_lives = {'fentanyl': 4.5, 'hydromorphone': 2.5, 'methadone': 19.5, 'morphine': 2.5,
                  'midazolam': 1.3, 'diazepam': 18, 'lorazepam': 14, 
                  'dexmedetomidine': 2, 'clonidine': 1.25, 
                  'ketamine': 2.5,
                  'propofol': 0.05,
                  'etomidate': 0.05,}

    return half_lives

def elimination_rates():
    """
    Calculate the elimination rates based on half-lives in min^{-1}.
    
    Returns:
        dict: A dictionary containing the elimination rates for each medication in min^{-1}
    """
    elimination_rates = {med: np.log(2) / (half_life * 60) for med, half_life in half_lives().items()}

    return elimination_rates

def filter_mar(mar):
    """
    Filter the MAR DataFrame.
    """

    mar['med_name'] = mar['med_name'].astype(str)
    mar['mar_action'] = mar['mar_action'].astype(str)

    agents = [
        'propofol', 'dexmedetomidine', 'midazolam', 'ketamine', 'diazepam',
        'lidocaine', 'clonidine', 'hydroxyzine', 'diphenhydramine', 'fentanyl',
        'hydromorphone', 'morphine', 'methadone', 'nalbuphine', 'acetaminophen']
    pattern = '|'.join(agents)

    mar = mar[mar['med_name'].str.lower().str.contains(pattern, regex=True, na=False)]
    mar = mar[~mar['mar_action'].str.contains('Missed', na=False)]
    mar = mar.dropna(subset=['dose'])

    groups = {
        "narcotics": ['fentanyl', 'morphine', 'hydromorphone', 'oxycodone', 'methadone', 'remifentanil'],
        "paralytics": ['rocuronium', 'vecuronium', 'succinylcholine', 'cisatracurium'],
        "alpha_agonists": ['dexmedetomidine', 'clonidine'],
        "ketamines": ['ketamine'],
        "propofols": ['propofol'],
        "etomidates": ['etomidate'],
        "benzodiazepines": ['midazolam', 'diazepam', 'lorazepam'],}

    results = []
    for _, drugs in groups.items():
        subpattern = '|'.join(drugs)
        results.append(
            mar[mar['med_name'].str.lower().str.contains(subpattern, regex=True, na=False)].reset_index(drop=True)
        )

    return tuple(results)


def filter_drug(mar, drug_name):
    mar = mar[mar['med_name'].str.lower().str.contains(drug_name, regex=True)].reset_index(drop=True)

    return mar

def calculate_doses(df):
    """
    Calculate the doses of continuous and bolus medications over time.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing time, dose, and mar_action columns.

    Returns:
        pd.DataFrame: DataFrame with time, dose, continuous_dose, and bolus_dose columns.
    """
    dose_df = pd.DataFrame({'time': pd.date_range(start=df['time'].iloc[0], end=df['time'].iloc[-1], freq='T')})

    doses = []
    continuous_doses = []
    bolus_doses = []

    index = 0
    continuous_dose = 0

    for i in range(len(dose_df)):
        bolus_dose = 0

        if dose_df['time'].iloc[i] == df['time'].iloc[index]:
            if 'Given' in df['mar_action'].iloc[index]:
                bolus_dose = df['dose'].iloc[index]
            else:
                continuous_dose = df['dose'].iloc[index] / 60
            index += 1
        
        rate = continuous_dose + bolus_dose
        doses.append(rate)
        continuous_doses.append(continuous_dose)
        bolus_doses.append(bolus_dose)

    dose_df['dose'] = doses
    dose_df['continuous_dose'] = continuous_doses
    dose_df['bolus_dose'] = bolus_doses

    return dose_df

def calculate_concentrations_euler(df, elimination_rate, start_time=None, end_time=None):
    """
    Calculate the drug concentration over time based on the elimination rate and dosing regimen.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing time and dose information.
        elimination_rate (float): The elimination rate of the drug.
        start_time (str): Start time for the calculation (optional).
        end_time (str): End time for the calculation (optional).
    
    Returns:
        pd.DataFrame: DataFrame with calculated drug concentrations over time.
    """
    if start_time is None:
        start_time = df['time'].iloc[0]
    if end_time is None:
        end_time = df['time'].iloc[-1]
    
    concentration_df = pd.DataFrame({'time': pd.date_range(start=start_time, end=end_time, freq='T')})
    concentration_df = concentration_df.merge(df[['time', 'dose', 'continuous_dose', 'bolus_dose']], on='time', how='left')
    concentration_df[['dose', 'continuous_dose', 'bolus_dose']] = concentration_df[['dose', 'continuous_dose', 'bolus_dose']].fillna(0)

    concentrations = [concentration_df['dose'].iloc[0]]

    for i in range(1, len(concentration_df)):
        concentration = concentration_df['dose'].iloc[i] + np.exp(-elimination_rate) * concentrations[-1]
        concentrations.append(concentration)

    concentration_df['concentration'] = concentrations

    return concentration_df

def calculate_concentrations_rk4(df, elimination_rate, start_time=None, end_time=None):
    """
    Calculate the drug concentration over time using the RK4 method.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing time and dose information.
        elimination_rate (float): The elimination rate of the drug.
        start_time (str): Start time for the calculation (optional).
        end_time (str): End time for the calculation (optional).
    
    Returns:
        pd.DataFrame: DataFrame with calculated drug concentrations over time.
    """
    if start_time is None:
        start_time = df['time'].iloc[0]
    if end_time is None:
        end_time = df['time'].iloc[-1]

    concentration_df = pd.DataFrame({'time': pd.date_range(start=start_time, end=end_time, freq='T')})
    concentration_df = concentration_df.merge(df[['time', 'dose', 'continuous_dose', 'bolus_dose']], on='time', how='left')
    concentration_df[['dose', 'continuous_dose', 'bolus_dose']] = concentration_df[['dose', 'continuous_dose', 'bolus_dose']].fillna(0)

    concentrations = [concentration_df['dose'].iloc[0]]

    for i in range(1, len(concentration_df)):
        dt = (concentration_df['time'].iloc[i] - concentration_df['time'].iloc[i-1]).total_seconds() / 60.0
        C = concentrations[-1]
        k1 = -elimination_rate * C + concentration_df['dose'].iloc[i]
        k2 = -elimination_rate * (C + 0.5 * dt * k1) + concentration_df['dose'].iloc[i]
        k3 = -elimination_rate * (C + 0.5 * dt * k2) + concentration_df['dose'].iloc[i]
        k4 = -elimination_rate * (C + dt * k3) + concentration_df['dose'].iloc[i]
        
        C_new = C + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        concentrations.append(C_new)

    concentration_df['concentration'] = concentrations

    return concentration_df

def calculate_concentrations_two_compartment_euler(df, weight, age, v1=None, v2=None, cl1=None, cl2=None, start_time=None, end_time=None):
    """
    Calculate the drug concentration over time using a two-compartment model with Euler's method. Model taken from Ginsberg et al. (1996).
    
    Parameters:
        df (pd.DataFrame): DataFrame containing time and dose information.
        cl1 (float): Clearance rate for the central compartment.
        cl2 (float): Clearance rate for the peripheral compartment.
        v1 (float): Volume of distribution for the central compartment.
        v2 (float): Volume of distribution for the peripheral compartment.
        start_time (str): Start time for the calculation (optional).
        end_time (str): End time for the calculation (optional).

    Returns:
        pd.DataFrame: DataFrame with calculated central and peripheral drug concentrations over time.
    """
    if v1 is None:
        v1 = (0.43 * (weight - 19.8) + 5.8)
    if v2 is None:
        v2 = (6.2 * (age - 6.4) + 34.4)
    if cl1 is None:
        cl1 = (0.01 * (weight - 19.8) + 0.35) / 60
    if cl2 is None:
        cl2 = (0.82) / 60

    if start_time is None:
        start_time = df['time'].iloc[0]
    if end_time is None:
        end_time = df['time'].iloc[-1]

    # if start_time < df['time'].iloc[0]:
    #     extra_times = pd.date_range(
    #         start=start_time,
    #         end=df['time'].iloc[0] - pd.Timedelta(seconds=1),
    #         freq='T'
    #     )
    #     padding = pd.DataFrame({
    #         'time': extra_times,
    #         'dose': 0,
    #         'continuous_dose': 0,
    #         'bolus_dose': 0
    #     })
    #     df = pd.concat([padding, df], ignore_index=True)
    # if end_time > df['time'].iloc[-1]:
    #     extra_times = pd.date_range(
    #         start=df['time'].iloc[-1] + pd.Timedelta(seconds=1),
    #         end=end_time,
    #         freq='T'
    #     )
    #     padding = pd.DataFrame({
    #         'time': extra_times,
    #         'dose': 0,
    #         'continuous_dose': 0,
    #         'bolus_dose': 0
    #     })
    #     df = pd.concat([df, padding], ignore_index=True)

    concentration_df = pd.DataFrame({'time': pd.date_range(start=start_time, end=end_time, freq='T')})
    concentration_df = concentration_df.merge(df[['time', 'dose', 'continuous_dose', 'bolus_dose']], on='time', how='left')
    concentration_df[['dose', 'continuous_dose', 'bolus_dose']] = concentration_df[['dose', 'continuous_dose', 'bolus_dose']].fillna(0)

    a1 = []
    a2 = []

    for i in range(len(concentration_df)):
        if i == 0:
            a1.append(concentration_df['dose'].iloc[i] * weight)
            a2.append(0)
        else:
            da1 = -cl1 * a1[-1] / v1 - cl2 * a1[-1] / v1 + cl2 * a2[-1] / v2 + concentration_df['continuous_dose'].iloc[i] * weight
            da2 = cl2 * a1[-1] / v1 - cl2 * a2[-1] / v2

            a1.append(a1[-1] + da1)
            a2.append(a2[-1] + da2)

    for i in range(len(a1)):
        a1[i] = a1[i] / weight
        a2[i] = a2[i] / weight
    
    concentration_df['concentration'] = a1
    concentration_df['concentration_peripheral'] = a2

    return concentration_df

def plot_concentration(drug_concentrations, drug_name='all', start=None, stop=None, show=False, save=None):
    """
    Plot the concentration of a drug over time.

    Parameters:
        drug_concentrations (dict): Dictionary containing DataFrames of drug concentrations.
        drug_name (list): List of drug names to plot.
        start (str): Start time for the plot. If None, the plot will start from the beginning of the data. Default is None.
        stop (str): End time for the plot. If None, the plot will end at the last time point of the data. Default is None.
        show (bool): Whether to show the plot. Default is True.
        save
        (str): File path to save the plot. If None, the plot will not be saved. Default is None.
    """
    drug_concentrations = drug_concentrations.copy()

    if not isinstance(drug_name, list):
        drug_name = [drug_name]

    if drug_name == ['all']:
        drug_name = drug_concentrations.keys()
        title = ''
    else:
        title = ', '.join(drug_name)

    if start is None:
        start = min([df['time'].iloc[0] for df in drug_concentrations.values()])
    
    if stop is None:
        stop = max([df['time'].iloc[-1] for df in drug_concentrations.values()])

    for drug in drug_concentrations:
        drug_concentrations[drug] = drug_concentrations[drug][(drug_concentrations[drug]['time'] >= pd.to_datetime(start)) & (drug_concentrations[drug]['time'] <= pd.to_datetime(stop))].reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    for drug in drug_name:
        plt.plot(drug_concentrations[drug]['time'], drug_concentrations[drug]['concentration'], label=drug)
    plt.xlabel('Time')
    plt.ylabel('Concentration (ug/kg)')

    if drug_name == drug_concentrations.keys():
        plt.title('Concentration over time')
    else:
        plt.title(f'Concentration of {title} over time')
    plt.legend()

    if save:
        plt.savefig(save)
    
    if show:
        plt.show()

    plt.close()

def plot_metrics(data, window=2, std=True, start=None, stop=None, show=False, save=None):
    """
    Plot various metrics over time.
    
    Parameters:
        data (dict): Dictionary containing DataFrames for each metric.
        window (int): Window size in seconds for moving average. Default is 2.
        std (bool): Whether to plot the standard deviation as a shaded area. Default is True.
        start (str): Start time for the plot. If None, the plot will start from the beginning of the data. Default is None.
        stop (str): End time for the plot. If None, the plot will end at the last time point of the data. Default is None.
        show (bool): Whether to show the plot. Default is True.
        save (str): File path to save the plot. If None, the plot will not be saved. Default is None.
    """
    data = data.copy()

    for metric in data:
        data[metric][metric] = data[metric][metric].rolling(window=window//2, min_periods=1).mean()
        
        if std:
            data[metric][f'{metric}_std'] = data[metric][metric].rolling(window=window//2, min_periods=1).std()

    _, axs = plt.subplots(len(data), 1, figsize=(10, 3*len(data)), sharex=True)

    for ax, metric in enumerate(data):
        if metric == "heart_rate":
            color = 'orange'
            ylabel = 'heart_rate (bpm)'
        elif metric == 'respiratory_rate':
            color = 'green'
            ylabel = 'respiratory_rate (bpm)'
        elif metric == 'acceleration':
            color = 'blue'
            ylabel = 'acceleration (g)'
        else:
            color = 'purple'
            ylabel = f'{metric}'

        if start is None:
            start = min([data[metric]['time'].iloc[0] for metric in data])
        
        if stop is None:
            stop = max([data[metric]['time'].iloc[-1] for metric in data])

        data[metric] = data[metric][(data[metric]['time'] >= pd.to_datetime(start)) & (data[metric]['time'] <= pd.to_datetime(stop))].reset_index(drop=True)

        datum = data[metric]
        axs[ax].plot(datum['time'], datum[f'{metric}'], label=f'{metric}_mean', color=color)

        if std:
            axs[ax].fill_between(datum['time'], 
                                 datum[f'{metric}'] - datum[f'{metric}_std'], 
                                 datum[f'{metric}'] + datum[f'{metric}_std'], 
                                 color=color, alpha=0.2, label=f'{metric} std')

        if window == 2:
            axs[ax].set_title(f'{metric} vs time')
        else:
            axs[ax].set_title(f'{metric} ({window}s avg) vs time')

        axs[ax].set_ylabel(ylabel)
        axs[ax].legend()
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save)

    if show:
        plt.show()
    
    plt.close()

def plot_metrics_and_concentrations(data, drug_concentrations, drug_name='all', window=2, std=True, start=None, stop=None, show=False, save=None):
    """
    Plot various metrics and drug concentration over time.
    
    Parameters:
        data (dict): Dictionary containing DataFrames for each metric.
        drug_concentrations (dict): Dictionary containing DataFrames of drug concentrations.
        drug_name (list): List of drug names to plot. Default is 'all'.
        window (int): Window size in seconds for moving average. Default is 2.
        std (bool): Whether to plot the standard deviation as a shaded area. Default is True.
        start (str): Start time for the plot. If None, the plot will start from the beginning of the data. Default is None.
        stop (str): End time for the plot. If None, the plot will end at the last time point of the data. Default is None.
        show (bool): Whether to show the plot. Default is True.
        save (str): File path to save the plot. If None, the plot will not be saved. Default is None.
    """
    data = data.copy()
    drug_concentrations = drug_concentrations.copy()

    for metric in data:
        data[metric][metric] = data[metric][metric].rolling(window=window//2, min_periods=1).mean()
        
        if std:
            data[metric][f'{metric}_std'] = data[metric][metric].rolling(window=window//2, min_periods=1).std()

    _, axs = plt.subplots(len(data)+1, 1, figsize=(10, 3*(len(data)+1)), sharex=True)

    for ax, metric in enumerate(data):
        if metric == "heart_rate":
            color = 'orange'
            ylabel = 'heart_rate (bpm)'
        elif metric == 'respiratory_rate':
            color = 'green'
            ylabel = 'respiratory_rate (bpm)'
        elif metric == 'acceleration':
            color = 'blue'
            ylabel = 'acceleration (g)'
        else:
            color = 'purple'
            ylabel = f'{metric}'

        if start is None:
            start = min([data[metric]['time'].iloc[0] for metric in data])
        
        if stop is None:
            stop = max([data[metric]['time'].iloc[-1] for metric in data])

        data[metric] = data[metric][(data[metric]['time'] >= pd.to_datetime(start)) & (data[metric]['time'] <= pd.to_datetime(stop))].reset_index(drop=True)

        datum = data[metric]
        axs[ax].plot(datum['time'], datum[f'{metric}'], label=f'{metric}_mean', color=color)

        if std:
            axs[ax].fill_between(datum['time'], 
                                 datum[f'{metric}'] - datum[f'{metric}_std'], 
                                 datum[f'{metric}'] + datum[f'{metric}_std'], 
                                 color=color, alpha=0.2, label=f'{metric} std')

        if window == 2:
            axs[ax].set_title(f'{metric} vs time')
        else:
            axs[ax].set_title(f'{metric} ({window}s avg) vs time')

        axs[ax].set_ylabel(ylabel)
        axs[ax].legend()

    axs[-1].set_title('Drug Concentration vs Time')

    if not isinstance(drug_name, list):
        drug_name = [drug_name]

    if drug_name == ['all']:
        drug_name = drug_concentrations.keys()
        title = ''
    else:
        title = ', '.join(drug_name)

    if start is None:
        start = min([df['time'].iloc[0] for df in drug_concentrations.values()])

    if stop is None:
        stop = max([df['time'].iloc[-1] for df in drug_concentrations.values()])
    
    for drug in drug_concentrations:
        drug_concentrations[drug] = drug_concentrations[drug][(drug_concentrations[drug]['time'] >= pd.to_datetime(start)) & (drug_concentrations[drug]['time'] <= pd.to_datetime(stop))].reset_index(drop=True)

    for drug in drug_name:
        if drug_concentrations[drug].empty:
            continue
        
        axs[-1].plot(drug_concentrations[drug]['time'], drug_concentrations[drug]['concentration'], label=drug)
        axs[-1].set_xlabel('Time')
        axs[-1].set_ylabel('Concentration (ug/kg)')
        axs[-1].legend()

        if drug_name == drug_concentrations.keys():
            axs[-1].set_title('Concentration over time')
        else:
            axs[-1].set_title(f'Concentration of {title} over time')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save)

    if show:
        plt.show()
    
    plt.close()

def main():
    pass

if __name__ == "__main__":
    main()