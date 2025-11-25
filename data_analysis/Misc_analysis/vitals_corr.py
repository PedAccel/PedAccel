import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
data_dir = 'data_analysis/Misc_analysis/PatientData'
output_dir = 'data_analysis/Misc_analysis/vitals_corr_results'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_vitals_data(patient_dir, patient_name):
    """
    Load vital signs data from the 1MIN_30MIN window .mat file.
    
    Parameters:
    patient_dir: Path to patient directory
    patient_name: Name of the patient
    
    Returns:
    dict: Vital signs data from .mat file
    """
    # Look for the 1MIN_30MIN window file
    vitals_file = os.path.join(patient_dir, f'{patient_name}_SICKBAY_1MIN_30MIN_Retro.mat')
    
    if not os.path.exists(vitals_file):
        print(f"No 1MIN_30MIN vitals file found for {patient_name}")
        print(f"Checked: {vitals_file}")
        return None
    
    try:
        vitals_data = loadmat(vitals_file)
        print(f"Available keys in {patient_name} vitals file: {list(vitals_data.keys())}")
        return vitals_data
    except Exception as e:
        print(f"Error loading vitals data for {patient_name}: {str(e)}")
        return None

def load_sbs_data(patient_dir, patient_name):
    """
    Load retrospective SBS scores.
    
    Parameters:
    patient_dir: Path to patient directory
    patient_name: Name of the patient
    
    Returns:
    DataFrame: SBS scores data
    """
    retro_file = os.path.join(patient_dir, f'{patient_name}_SBS_Scores_Retro.xlsx')
    
    if not os.path.exists(retro_file):
        print(f"No retrospective SBS file found for {patient_name}")
        return None
    
    try:
        df = pd.read_excel(retro_file)
        df['Time_uniform'] = pd.to_datetime(df['Time_uniform'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        df['SBS_numeric'] = pd.to_numeric(df['SBS'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading SBS data for {patient_name}: {str(e)}")
        return None

def extract_first_15_minutes(vitals_data, sbs_data):
    """
    Extract the first 15 minutes of vital signs for each SBS event.
    
    Parameters:
    vitals_data: Vital signs data from .mat file
    sbs_data: SBS scores data
    
    Returns:
    list: List of (SBS_score, hr_15min, rr_15min) tuples
    """
    results = []
    
    # Get vital signs arrays
    hr_data = vitals_data.get('heart_rate', None)
    rr_data = vitals_data.get('respiratory_rate', None)
    sbs_scores = vitals_data.get('sbs', None)
    
    if hr_data is None or rr_data is None or sbs_scores is None:
        print("Missing vital signs data in .mat file")
        return results
    
    print(f"Data shapes - HR: {hr_data.shape}, RR: {rr_data.shape}, SBS: {sbs_scores.shape}")
    
    # Get the correct number of SBS events
    if len(sbs_scores.shape) == 2:
        n_events = sbs_scores.shape[1]  # If SBS is (1, n), use n
    else:
        n_events = len(sbs_scores)
    
    print(f"Number of SBS events: {n_events}")
    
    # Process each SBS event
    for i in range(n_events):
        # Extract SBS score correctly based on array shape
        if len(sbs_scores.shape) == 2:
            sbs_score = sbs_scores[0, i]  # If SBS is (1, n), use [0, i]
        else:
            sbs_score = sbs_scores[i]
        
        # Get first 15 minutes of vital signs (assuming 0.5 Hz sampling = 450 samples for 15 min)
        # But let's be conservative and take first 450 samples or all if less
        if i < hr_data.shape[0] and i < rr_data.shape[0]:
            hr_window = hr_data[i]
            rr_window = rr_data[i]
            
            if i < 5:  # Only print first 5 windows to avoid spam
                print(f"Window {i} - HR shape: {hr_window.shape}, RR shape: {rr_window.shape}")
            
            # Take first 15 minutes (450 samples at 0.5 Hz)
            n_samples_15min = min(450, len(hr_window))
            
            if n_samples_15min > 0:
                hr_15min = hr_window[:n_samples_15min]
                rr_15min = rr_window[:n_samples_15min]
                
                # Calculate mean values for the 15-minute window
                hr_mean = np.mean(hr_15min) if len(hr_15min) > 0 else np.nan
                rr_mean = np.mean(rr_15min) if len(rr_15min) > 0 else np.nan
                
                if i < 5:  # Only print first 5 windows
                    print(f"  SBS: {sbs_score}, HR mean: {hr_mean}, RR mean: {rr_mean}")
                
                # Only include if we have valid data
                if not (np.isnan(hr_mean) or np.isnan(rr_mean) or np.isnan(sbs_score)):
                    results.append((sbs_score, hr_mean, rr_mean))
                    if i < 5:  # Only print first 5 windows
                        print(f"  -> Added to results")
                else:
                    if i < 5:  # Only print first 5 windows
                        print(f"  -> Skipped due to NaN values")
    
    print(f"Total valid results: {len(results)}")
    return results

def calculate_vitals_correlations(patient_data, patient_name):
    """
    Calculate correlations between SBS scores and vital signs.
    
    Parameters:
    patient_data: List of (SBS_score, hr_mean, rr_mean) tuples
    patient_name: Name of the patient
    
    Returns:
    dict: Correlation metrics
    """
    if len(patient_data) < 5:  # Need at least 5 data points
        print(f"Insufficient data for {patient_name}: {len(patient_data)} valid points")
        return None
    
    # Extract data
    sbs_scores = [data[0] for data in patient_data]
    hr_means = [data[1] for data in patient_data]
    rr_means = [data[2] for data in patient_data]
    
    metrics = {}
    
    # SBS vs Heart Rate correlations
    try:
        metrics['sbs_hr_pearson_r'], metrics['sbs_hr_pearson_p'] = pearsonr(sbs_scores, hr_means)
    except:
        metrics['sbs_hr_pearson_r'] = metrics['sbs_hr_pearson_p'] = np.nan
    
    try:
        metrics['sbs_hr_spearman_r'], metrics['sbs_hr_spearman_p'] = spearmanr(sbs_scores, hr_means)
    except:
        metrics['sbs_hr_spearman_r'] = metrics['sbs_hr_spearman_p'] = np.nan
    
    # SBS vs Respiratory Rate correlations
    try:
        metrics['sbs_rr_pearson_r'], metrics['sbs_rr_pearson_p'] = pearsonr(sbs_scores, rr_means)
    except:
        metrics['sbs_rr_pearson_r'] = metrics['sbs_rr_pearson_p'] = np.nan
    
    try:
        metrics['sbs_rr_spearman_r'], metrics['sbs_rr_spearman_p'] = spearmanr(sbs_scores, rr_means)
    except:
        metrics['sbs_rr_spearman_r'] = metrics['sbs_rr_spearman_p'] = np.nan
    
    # Vital signs correlation
    try:
        metrics['hr_rr_pearson_r'], metrics['hr_rr_pearson_p'] = pearsonr(hr_means, rr_means)
    except:
        metrics['hr_rr_pearson_r'] = metrics['hr_rr_pearson_p'] = np.nan
    
    # Summary statistics
    metrics['n_events'] = len(patient_data)
    metrics['mean_sbs'] = np.mean(sbs_scores)
    metrics['std_sbs'] = np.std(sbs_scores)
    metrics['mean_hr'] = np.mean(hr_means)
    metrics['std_hr'] = np.std(hr_means)
    metrics['mean_rr'] = np.mean(rr_means)
    metrics['std_rr'] = np.std(rr_means)
    
    # Patient info
    metrics['patient'] = patient_name
    
    return metrics

def create_vitals_correlation_plots(all_patient_metrics, output_dir):
    """
    Create plots for vital signs correlation analysis.
    
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
    fig.suptitle('SBS-Vital Signs Correlation Analysis (First 15 min)', fontsize=16)
    
    # Plot 1: SBS vs Heart Rate correlation by patient
    patients = [m['patient'] for m in valid_metrics]
    sbs_hr_pearson = [m['sbs_hr_pearson_r'] for m in valid_metrics]
    sbs_hr_spearman = [m['sbs_hr_spearman_r'] for m in valid_metrics]
    
    x = np.arange(len(patients))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, sbs_hr_pearson, width, label='Pearson r', alpha=0.7)
    axes[0, 0].bar(x + width/2, sbs_hr_spearman, width, label='Spearman r', alpha=0.7)
    axes[0, 0].set_xlabel('Patient')
    axes[0, 0].set_ylabel('Correlation Coefficient')
    axes[0, 0].set_title('SBS vs Heart Rate Correlation')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(patients, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: SBS vs Respiratory Rate correlation by patient
    sbs_rr_pearson = [m['sbs_rr_pearson_r'] for m in valid_metrics]
    sbs_rr_spearman = [m['sbs_rr_spearman_r'] for m in valid_metrics]
    
    axes[0, 1].bar(x - width/2, sbs_rr_pearson, width, label='Pearson r', alpha=0.7)
    axes[0, 1].bar(x + width/2, sbs_rr_spearman, width, label='Spearman r', alpha=0.7)
    axes[0, 1].set_xlabel('Patient')
    axes[0, 1].set_ylabel('Correlation Coefficient')
    axes[0, 1].set_title('SBS vs Respiratory Rate Correlation')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(patients, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Mean vital signs by patient
    mean_hr = [m['mean_hr'] for m in valid_metrics]
    mean_rr = [m['mean_rr'] for m in valid_metrics]
    
    axes[1, 0].bar(x - width/2, mean_hr, width, label='Mean HR', alpha=0.7)
    axes[1, 0].bar(x + width/2, mean_rr, width, label='Mean RR', alpha=0.7)
    axes[1, 0].set_xlabel('Patient')
    axes[1, 0].set_ylabel('Vital Sign Value')
    axes[1, 0].set_title('Mean Vital Signs by Patient')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(patients, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Number of events by patient
    n_events = [m['n_events'] for m in valid_metrics]
    
    axes[1, 1].bar(patients, n_events, alpha=0.7)
    axes[1, 1].set_xlabel('Patient')
    axes[1, 1].set_ylabel('Number of Events')
    axes[1, 1].set_title('Number of Events by Patient')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vitals_correlation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_population_metrics(all_patient_metrics):
    """
    Calculate population-level vital signs correlation metrics.
    
    Parameters:
    all_patient_metrics: List of patient-level metrics
    
    Returns:
    dict: Population-level metrics
    """
    if not all_patient_metrics:
        return None
    
    # Combine all SBS and vital signs data across patients
    all_sbs = []
    all_hr = []
    all_rr = []
    
    for patient_metrics in all_patient_metrics:
        if patient_metrics is None:
            continue
            
        # We need to reload the data to get the actual scores
        patient_name = patient_metrics['patient']
        patient_dir = os.path.join('data_analysis/Misc_analysis/PatientData', patient_name)
        
        # Load vitals data
        vitals_data = load_vitals_data(patient_dir, patient_name)
        sbs_data = load_sbs_data(patient_dir, patient_name)
        
        if vitals_data is not None and sbs_data is not None:
            patient_data = extract_first_15_minutes(vitals_data, sbs_data)
            
            for sbs_score, hr_mean, rr_mean in patient_data:
                all_sbs.append(sbs_score)
                all_hr.append(hr_mean)
                all_rr.append(rr_mean)
    
    if len(all_sbs) < 10:
        print("Insufficient data for population analysis")
        return None
    
    # Calculate population metrics
    try:
        sbs_hr_pearson_r, sbs_hr_pearson_p = pearsonr(all_sbs, all_hr)
    except:
        sbs_hr_pearson_r = sbs_hr_pearson_p = np.nan
    
    try:
        sbs_hr_spearman_r, sbs_hr_spearman_p = spearmanr(all_sbs, all_hr)
    except:
        sbs_hr_spearman_r = sbs_hr_spearman_p = np.nan
    
    try:
        sbs_rr_pearson_r, sbs_rr_pearson_p = pearsonr(all_sbs, all_rr)
    except:
        sbs_rr_pearson_r = sbs_rr_pearson_p = np.nan
    
    try:
        sbs_rr_spearman_r, sbs_rr_spearman_p = spearmanr(all_sbs, all_rr)
    except:
        sbs_rr_spearman_r = sbs_rr_spearman_p = np.nan
    
    population_metrics = {
        'patient': 'POPULATION',
        'sbs_hr_pearson_r': sbs_hr_pearson_r,
        'sbs_hr_pearson_p': sbs_hr_pearson_p,
        'sbs_hr_spearman_r': sbs_hr_spearman_r,
        'sbs_hr_spearman_p': sbs_hr_spearman_p,
        'sbs_rr_pearson_r': sbs_rr_pearson_r,
        'sbs_rr_pearson_p': sbs_rr_pearson_p,
        'sbs_rr_spearman_r': sbs_rr_spearman_r,
        'sbs_rr_spearman_p': sbs_rr_spearman_p,
        'total_events': len(all_sbs),
        'mean_sbs': np.mean(all_sbs),
        'std_sbs': np.std(all_sbs),
        'mean_hr': np.mean(all_hr),
        'std_hr': np.std(all_hr),
        'mean_rr': np.mean(all_rr),
        'std_rr': np.std(all_rr),
        'n_patients': len([m for m in all_patient_metrics if m is not None])
    }
    
    return population_metrics

def main():
    """
    Main function to analyze SBS-vital signs correlations.
    """
    print("Starting SBS-Vital Signs correlation analysis...")
    
    # Get list of specific patient directories
    all_patient_dirs = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d)) and 'Patient' in d]
    
    # Filter to only the specified patients
    target_patients = ['Patient3', 'Patient4', 'Patient9', 'Patient11', 'Patient14']
    patient_dirs = [d for d in all_patient_dirs if d in target_patients]
    patient_dirs.sort()
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Process each patient
    all_patient_metrics = []
    
    for patient_dir in patient_dirs:
        patient_path = os.path.join(data_dir, patient_dir)
        print(f"\nProcessing {patient_dir}...")
        
        # Load vitals data
        vitals_data = load_vitals_data(patient_path, patient_dir)
        sbs_data = load_sbs_data(patient_path, patient_dir)
        
        if vitals_data is not None and sbs_data is not None:
            # Extract first 15 minutes of vital signs
            patient_data = extract_first_15_minutes(vitals_data, sbs_data)
            
            # Calculate correlations
            metrics = calculate_vitals_correlations(patient_data, patient_dir)
            all_patient_metrics.append(metrics)
            
            if metrics:
                print(f"  Total events: {metrics['n_events']}")
                print(f"  Mean SBS: {metrics['mean_sbs']:.2f}")
                print(f"  Mean HR: {metrics['mean_hr']:.1f}")
                print(f"  Mean RR: {metrics['mean_rr']:.1f}")
                print(f"  SBS-HR Pearson r: {metrics['sbs_hr_pearson_r']:.3f}")
                print(f"  SBS-RR Pearson r: {metrics['sbs_rr_pearson_r']:.3f}")
                print(f"  SBS-HR Spearman r: {metrics['sbs_hr_spearman_r']:.3f}")
                print(f"  SBS-RR Spearman r: {metrics['sbs_rr_spearman_r']:.3f}")
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
        print(f"  Mean SBS: {population_metrics['mean_sbs']:.2f}")
        print(f"  Mean HR: {population_metrics['mean_hr']:.1f}")
        print(f"  Mean RR: {population_metrics['mean_rr']:.1f}")
        print(f"  SBS-HR Pearson r: {population_metrics['sbs_hr_pearson_r']:.3f}")
        print(f"  SBS-RR Pearson r: {population_metrics['sbs_rr_pearson_r']:.3f}")
        print(f"  SBS-HR Spearman r: {population_metrics['sbs_hr_spearman_r']:.3f}")
        print(f"  SBS-RR Spearman r: {population_metrics['sbs_rr_spearman_r']:.3f}")
    
    # Create summary plots
    print(f"\nCreating summary plots...")
    create_vitals_correlation_plots(all_patient_metrics, output_dir)
    
    # Save results to CSV
    print(f"\nSaving results...")
    
    # Patient-level results
    valid_patient_metrics = [m for m in all_patient_metrics if m is not None]
    if valid_patient_metrics:
        patient_df = pd.DataFrame(valid_patient_metrics)
        patient_df.to_csv(os.path.join(output_dir, 'patient_vitals_metrics.csv'), index=False)
        print(f"Patient-level results saved to: {os.path.join(output_dir, 'patient_vitals_metrics.csv')}")
    
    # Population-level results
    if population_metrics:
        population_df = pd.DataFrame([population_metrics])
        population_df.to_csv(os.path.join(output_dir, 'population_vitals_metrics.csv'), index=False)
        print(f"Population-level results saved to: {os.path.join(output_dir, 'population_vitals_metrics.csv')}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("Summary plots:")
    print(f"  - vitals_correlation_plots.png")

if __name__ == "__main__":
    main()
