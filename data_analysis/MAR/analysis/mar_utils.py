import numpy as np
import pandas as pd

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
