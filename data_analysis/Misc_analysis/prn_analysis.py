'''
PRN-SBS Correlation Analysis
|_ Sidharth Raghavan 08/04/2025
'''

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set up paths
data_dir = 'data_analysis/Misc_analysis/PatientData'
output_dir = 'data_analysis/Misc_analysis/prn_results'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_patient_data(patient_dir, patient_name):
    """
    Load SBS and PRN data for a single patient.
    
    Parameters:
    patient_dir: Path to patient directory
    patient_name: Name of the patient
    
    Returns:
    DataFrame: Combined SBS and PRN data
    """
    retro_file = os.path.join(patient_dir, f'{patient_name}_SBS_Scores_Retro.xlsx')
    
    if not os.path.exists(retro_file):
        print(f"No retrospective file found for {patient_name}")
        return None
    
    try:
        # Load the Excel file
        df = pd.read_excel(retro_file)
        
        # Check if required columns exist
        required_cols = ['SBS', 'SedPRN_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing columns for {patient_name}: {missing_cols}")
            return None
        
        # Convert timestamps
        df['Time_uniform'] = pd.to_datetime(df['Time_uniform'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        
        # Convert SBS to numeric
        df['SBS_numeric'] = pd.to_numeric(df['SBS'], errors='coerce')
        
        # Create PRN indicator (any PRN given)
        df['PRN_given'] = df['SedPRN_type'].notna() & (df['SedPRN_type'] != '')
        
        # Convert PRN to numeric (1 if given, 0 if not)
        df['PRN_numeric'] = df['PRN_given'].astype(int)
        
        return df
        
    except Exception as e:
        print(f"Error processing {patient_name}: {str(e)}")
        return None

def analyze_prn_sbs_correlation(df, patient_name):
    """
    Analyze correlation between SBS scores and PRN administration.
    
    Parameters:
    df: DataFrame with SBS and PRN data
    patient_name: Name of the patient
    
    Returns:
    dict: Correlation metrics
    """
    # Remove rows with missing data
    valid_mask = ~(df['SBS_numeric'].isna() | df['PRN_numeric'].isna())
    df_clean = df[valid_mask]
    
    if len(df_clean) < 10:  # Need at least 10 data points
        print(f"Insufficient data for {patient_name}: {len(df_clean)} valid points")
        return None
    
    sbs_scores = df_clean['SBS_numeric']
    prn_given = df_clean['PRN_numeric']
    
    metrics = {}
    
    # Correlation coefficients
    try:
        metrics['pearson_r'], metrics['pearson_p'] = pearsonr(sbs_scores, prn_given)
    except:
        metrics['pearson_r'] = metrics['pearson_p'] = np.nan
    
    try:
        metrics['spearman_r'], metrics['spearman_p'] = spearmanr(sbs_scores, prn_given)
    except:
        metrics['spearman_r'] = metrics['spearman_p'] = np.nan
    
    # PRN analysis by SBS score
    sbs_prn_counts = df_clean.groupby('SBS_numeric')['PRN_numeric'].agg(['count', 'sum', 'mean']).reset_index()
    sbs_prn_counts.columns = ['SBS_score', 'total_events', 'prn_given', 'prn_rate']
    
    # Overall statistics
    metrics['total_events'] = len(df_clean)
    metrics['total_prn'] = prn_given.sum()
    metrics['overall_prn_rate'] = prn_given.mean()
    metrics['mean_sbs'] = sbs_scores.mean()
    metrics['std_sbs'] = sbs_scores.std()
    
    # SBS-specific PRN rates
    metrics['sbs_prn_rates'] = sbs_prn_counts.to_dict('records')
    
    # Patient info
    metrics['patient'] = patient_name
    
    return metrics

def create_prn_analysis_plots(all_patient_metrics, output_dir):
    """
    Create plots for PRN-SBS correlation analysis.
    
    Parameters:
    all_patient_metrics: List of patient-level metrics
    output_dir: Directory to save plots
    """
    if not all_patient_metrics:
        print("No valid metrics to plot")
        return
    
    # Filter out None values
    valid_metrics = [m for m in all_patient_metrics if m is not None]
    
    if not valid_metrics:
        print("No valid metrics to plot")
        return
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PRN-SBS Correlation Analysis', fontsize=16)
    
    # Plot 1: Pearson correlation by patient
    patients = [m['patient'] for m in valid_metrics]
    pearson_r = [m['pearson_r'] for m in valid_metrics]
    spearman_r = [m['spearman_r'] for m in valid_metrics]
    
    x = np.arange(len(patients))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, pearson_r, width, label='Pearson r', alpha=0.7)
    axes[0, 0].bar(x + width/2, spearman_r, width, label='Spearman r', alpha=0.7)
    axes[0, 0].set_xlabel('Patient')
    axes[0, 0].set_ylabel('Correlation Coefficient')
    axes[0, 0].set_title('PRN-SBS Correlation by Patient')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(patients, rotation=45)
    axes[0, 0].legend()
    
    # Plot 2: Overall PRN rate by patient
    overall_prn_rates = [m['overall_prn_rate'] for m in valid_metrics]
    
    axes[0, 1].bar(patients, overall_prn_rates, alpha=0.7)
    axes[0, 1].set_xlabel('Patient')
    axes[0, 1].set_ylabel('Overall PRN Rate')
    axes[0, 1].set_title('Overall PRN Rate by Patient')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Mean SBS by patient
    mean_sbs = [m['mean_sbs'] for m in valid_metrics]
    
    axes[1, 0].bar(patients, mean_sbs, alpha=0.7)
    axes[1, 0].set_xlabel('Patient')
    axes[1, 0].set_ylabel('Mean SBS Score')
    axes[1, 0].set_title('Mean SBS Score by Patient')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Total events vs PRN given
    total_events = [m['total_events'] for m in valid_metrics]
    total_prn = [m['total_prn'] for m in valid_metrics]
    
    axes[1, 1].scatter(total_events, total_prn, alpha=0.7, s=100)
    axes[1, 1].set_xlabel('Total Events')
    axes[1, 1].set_ylabel('Total PRN Given')
    axes[1, 1].set_title('Total Events vs PRN Given')
    
    # Add patient labels to scatter plot
    for i, patient in enumerate(patients):
        axes[1, 1].annotate(patient, (total_events[i], total_prn[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prn_sbs_correlation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create SBS-specific PRN rate plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Combine all SBS scores and their PRN rates
    all_sbs_scores = []
    all_prn_rates = []
    
    for metrics in valid_metrics:
        for sbs_data in metrics['sbs_prn_rates']:
            all_sbs_scores.append(sbs_data['SBS_score'])
            all_prn_rates.append(sbs_data['prn_rate'])
    
    # Create box plot
    sbs_unique = sorted(set(all_sbs_scores))
    prn_rates_by_sbs = []
    sbs_labels = []
    
    for sbs in sbs_unique:
        rates = [rate for score, rate in zip(all_sbs_scores, all_prn_rates) if score == sbs]
        if rates:
            prn_rates_by_sbs.append(rates)
            sbs_labels.append(f'SBS {sbs}')
    
    if prn_rates_by_sbs:
        ax.boxplot(prn_rates_by_sbs, labels=sbs_labels)
        ax.set_xlabel('SBS Score')
        ax.set_ylabel('PRN Rate')
        ax.set_title('PRN Rate by SBS Score (Across All Patients)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prn_rate_by_sbs_score.png'), dpi=300, bbox_inches='tight')
        plt.close()

def calculate_population_metrics(all_patient_metrics):
    """
    Calculate population-level PRN-SBS correlation metrics.
    
    Parameters:
    all_patient_metrics: List of patient-level metrics
    
    Returns:
    dict: Population-level metrics
    """
    if not all_patient_metrics:
        return None
    
    # Combine all SBS and PRN data across patients
    all_sbs = []
    all_prn = []
    
    for patient_metrics in all_patient_metrics:
        if patient_metrics is None:
            continue
            
        # We need to reload the data to get the actual scores
        patient_name = patient_metrics['patient']
        patient_dir = os.path.join('data_analysis/Misc_analysis/PatientData', patient_name)
        df = load_patient_data(patient_dir, patient_name)
        
        if df is not None:
            valid_mask = ~(df['SBS_numeric'].isna() | df['PRN_numeric'].isna())
            df_clean = df[valid_mask]
            
            all_sbs.extend(df_clean['SBS_numeric'].tolist())
            all_prn.extend(df_clean['PRN_numeric'].tolist())
    
    if len(all_sbs) < 10:
        print("Insufficient data for population analysis")
        return None
    
    # Calculate population metrics
    try:
        pearson_r, pearson_p = pearsonr(all_sbs, all_prn)
    except:
        pearson_r = pearson_p = np.nan
    
    try:
        spearman_r, spearman_p = spearmanr(all_sbs, all_prn)
    except:
        spearman_r = spearman_p = np.nan
    
    population_metrics = {
        'patient': 'POPULATION',
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'total_events': len(all_sbs),
        'total_prn': sum(all_prn),
        'overall_prn_rate': np.mean(all_prn),
        'mean_sbs': np.mean(all_sbs),
        'std_sbs': np.std(all_sbs),
        'n_patients': len([m for m in all_patient_metrics if m is not None])
    }
    
    return population_metrics

def main():
    """
    Main function to analyze PRN-SBS correlations across all patients.
    """
    print("Starting PRN-SBS correlation analysis...")
    
    # Get list of patient directories
    patient_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and 'Patient' in d]
    patient_dirs.sort()
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Process each patient
    all_patient_metrics = []
    
    for patient_dir in patient_dirs:
        patient_path = os.path.join(data_dir, patient_dir)
        print(f"\nProcessing {patient_dir}...")
        
        # Load patient data
        df = load_patient_data(patient_path, patient_dir)
        
        if df is not None:
            # Analyze correlations
            metrics = analyze_prn_sbs_correlation(df, patient_dir)
            all_patient_metrics.append(metrics)
            
            if metrics:
                print(f"  Total events: {metrics['total_events']}")
                print(f"  Total PRN given: {metrics['total_prn']}")
                print(f"  Overall PRN rate: {metrics['overall_prn_rate']:.3f}")
                print(f"  Mean SBS: {metrics['mean_sbs']:.2f}")
                print(f"  Pearson r: {metrics['pearson_r']:.3f}")
                print(f"  Spearman r: {metrics['spearman_r']:.3f}")
                
                # Print SBS-specific PRN rates
                print(f"  PRN rates by SBS score:")
                for sbs_data in metrics['sbs_prn_rates']:
                    print(f"    SBS {sbs_data['SBS_score']}: {sbs_data['prn_rate']:.3f} ({sbs_data['prn_given']}/{sbs_data['total_events']})")
            else:
                print(f"  No valid data found")
        else:
            print(f"  Could not load data")
            all_patient_metrics.append(None)
    
    # Calculate population metrics
    print(f"\nCalculating population-level metrics...")
    population_metrics = calculate_population_metrics(all_patient_metrics)
    
    if population_metrics:
        print(f"Population-level results:")
        print(f"  Total patients: {population_metrics['n_patients']}")
        print(f"  Total events: {population_metrics['total_events']}")
        print(f"  Total PRN given: {population_metrics['total_prn']}")
        print(f"  Overall PRN rate: {population_metrics['overall_prn_rate']:.3f}")
        print(f"  Mean SBS: {population_metrics['mean_sbs']:.2f}")
        print(f"  Pearson r: {population_metrics['pearson_r']:.3f}")
        print(f"  Spearman r: {population_metrics['spearman_r']:.3f}")
    
    # Create summary plots
    print(f"\nCreating summary plots...")
    create_prn_analysis_plots(all_patient_metrics, output_dir)
    
    # Save results to CSV
    print(f"\nSaving results...")
    
    # Patient-level results
    valid_patient_metrics = [m for m in all_patient_metrics if m is not None]
    if valid_patient_metrics:
        # Flatten the nested SBS-PRN rates for CSV
        flattened_metrics = []
        for metrics in valid_patient_metrics:
            base_metrics = {k: v for k, v in metrics.items() if k != 'sbs_prn_rates'}
            flattened_metrics.append(base_metrics)
        
        patient_df = pd.DataFrame(flattened_metrics)
        patient_df.to_csv(os.path.join(output_dir, 'patient_prn_metrics.csv'), index=False)
        print(f"Patient-level results saved to: {os.path.join(output_dir, 'patient_prn_metrics.csv')}")
    
    # Population-level results
    if population_metrics:
        population_df = pd.DataFrame([population_metrics])
        population_df.to_csv(os.path.join(output_dir, 'population_prn_metrics.csv'), index=False)
        print(f"Population-level results saved to: {os.path.join(output_dir, 'population_prn_metrics.csv')}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("Summary plots:")
    print(f"  - prn_sbs_correlation_plots.png")
    print(f"  - prn_rate_by_sbs_score.png")

if __name__ == "__main__":
    main()
