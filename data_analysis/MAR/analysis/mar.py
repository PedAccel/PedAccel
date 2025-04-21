import numpy as np
import pandas as pd

def classes():
    """
    Define the classes of medications used in the analysis.

    Returns:
        pd.DataFrame: DataFrame containing the classes of medications.
    """
    narcotics = ['fentanyl', 'morphine', 'hydromorphone', 'oxycodone', 'methadone', 'remifentanil']
    paralytics = ['rocuronium', 'vecuronium', 'succinylcholine', 'cisatracurium']
    alpha_agonists = ['dexmedetomidine', 'clonidine']
    ketamines = ['ketamine']
    propofols = ['propofol']
    etomidates = ['etomidate']
    benzodiazepines = ['midazolam', 'diazepam', 'lorazepam']

    classes = pd.DataFrame({
        'narcotics': narcotics,
        'paralytics': paralytics,
        'alpha_agonists': alpha_agonists,
        'ketamines': ketamines,
        'propofols': propofols,
        'etomidates': etomidates,
        'benzodiazepines': benzodiazepines
    })

    return classes

def filter_mar(mar):
    """
    Filter the MAR DataFrame.
    
    Parameters:
        mar_df (pd.DataFrame): The MAR DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered narcotics.
        pd.DataFrame: Filtered paralytics.
        pd.DataFrame: Filtered alpha_agonists.
        pd.DataFrame: Filtered ketamines.
        pd.DataFrame: Filtered propofols.
        pd.DataFrame: Filtered etomidates.
        pd.DataFrame: Filtered benzodiazepines.
    """

    agents = ['propofol', 'dexmedetomidine', 'midazolam', 'ketamine', 'diazepam', 'lidocaine', 'clonidine', 'hydroxyzine', 'diphenhydramine', 'fentanyl', 'hydromorphone', 'morphine', 'methadone', 'nalbuphine', 'acetaminophen']
    pattern = '|'.join(agents)
    mar = mar[mar['med_name'].str.lower().str.contains(pattern, regex=True)]
    mar = mar[~mar['mar_action'].str.contains('Missed')]
    mar = mar.dropna(subset=['dose'])

    narcotics = ['fentanyl', 'morphine', 'hydromorphone', 'oxycodone', 'methadone', 'remifentanil']
    paralytics = ['rocuronium', 'vecuronium', 'succinylcholine', 'cisatracurium']
    alpha_agonists = ['dexmedetomidine', 'clonidine']
    ketamines = ['ketamine']
    propofols = ['propofol']
    etomidates = ['etomidate']
    benzodiazepines = ['midazolam', 'diazepam', 'lorazepam']

    narcotics_pattern = '|'.join(narcotics)
    paralytics_pattern = '|'.join(paralytics)
    alpha_agonists_pattern = '|'.join(alpha_agonists)
    ketamines_pattern = '|'.join(ketamines)
    propofols_pattern = '|'.join(propofols)
    etomidates_pattern = '|'.join(etomidates)
    benzodiazepines_pattern = '|'.join(benzodiazepines)

    mar_narcotics = mar[mar['med_name'].str.lower().str.contains(narcotics_pattern, regex=True)].reset_index(drop=True)
    mar_paralytics = mar[mar['med_name'].str.lower().str.contains(paralytics_pattern, regex=True)].reset_index(drop=True)
    mar_alpha_agonists = mar[mar['med_name'].str.lower().str.contains(alpha_agonists_pattern, regex=True)].reset_index(drop=True)
    mar_ketamines = mar[mar['med_name'].str.lower().str.contains(ketamines_pattern, regex=True)].reset_index(drop=True)
    mar_propofols = mar[mar['med_name'].str.lower().str.contains(propofols_pattern, regex=True)].reset_index(drop=True)
    mar_etomidates = mar[mar['med_name'].str.lower().str.contains(etomidates_pattern, regex=True)].reset_index(drop=True)
    mar_benzodiazepines = mar[mar['med_name'].str.lower().str.contains(benzodiazepines_pattern, regex=True)].reset_index(drop=True)

    return mar_narcotics, mar_paralytics, mar_alpha_agonists, mar_ketamines, mar_propofols, mar_etomidates, mar_benzodiazepines

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

def calculate_concentrations(df, elimination_rate, start_time=None, end_time=None):
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

def main():
    pass

if __name__ == "__main__":
    main()