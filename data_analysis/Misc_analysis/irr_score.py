import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_irr_metrics(rater1_scores, rater2_scores, ground_truth_scores):
    """
    Calculate interrater reliability metrics between two raters and ground truth.
    
    Parameters:
    rater1_scores: Series of scores from first rater
    rater2_scores: Series of scores from second rater  
    ground_truth_scores: Series of verified ground truth scores
    
    Returns:
    dict: Dictionary containing all IRR metrics
    """
    # Remove any NaN values
    valid_mask = ~(rater1_scores.isna() | rater2_scores.isna() | ground_truth_scores.isna())
    
    if valid_mask.sum() < 2:  # Need at least 2 valid comparisons
        return None
    
    r1_clean = rater1_scores[valid_mask]
    r2_clean = rater2_scores[valid_mask]
    gt_clean = ground_truth_scores[valid_mask]
    
    metrics = {}
    
    # Correlation coefficients
    try:
        metrics['rater1_pearson_r'], metrics['rater1_pearson_p'] = pearsonr(r1_clean, gt_clean)
    except:
        metrics['rater1_pearson_r'] = metrics['rater1_pearson_p'] = np.nan
    
    try:
        metrics['rater2_pearson_r'], metrics['rater2_pearson_p'] = pearsonr(r2_clean, gt_clean)
    except:
        metrics['rater2_pearson_r'] = metrics['rater2_pearson_p'] = np.nan
    
    try:
        metrics['rater1_spearman_r'], metrics['rater1_spearman_p'] = spearmanr(r1_clean, gt_clean)
    except:
        metrics['rater1_spearman_r'] = metrics['rater1_spearman_p'] = np.nan
    
    try:
        metrics['rater2_spearman_r'], metrics['rater2_spearman_p'] = spearmanr(r2_clean, gt_clean)
    except:
        metrics['rater2_spearman_r'] = metrics['rater2_spearman_p'] = np.nan
    
    # Error metrics
    metrics['rater1_mae'] = mean_absolute_error(gt_clean, r1_clean)
    metrics['rater2_mae'] = mean_absolute_error(gt_clean, r2_clean)
    metrics['rater1_rmse'] = np.sqrt(mean_squared_error(gt_clean, r1_clean))
    metrics['rater2_rmse'] = np.sqrt(mean_squared_error(gt_clean, r2_clean))
    
    # Cohen's Kappa (for categorical agreement)
    try:
        metrics['rater1_kappa'] = cohen_kappa_score(gt_clean, r1_clean)
    except:
        metrics['rater1_kappa'] = np.nan
    
    try:
        metrics['rater2_kappa'] = cohen_kappa_score(gt_clean, r2_clean)
    except:
        metrics['rater2_kappa'] = np.nan
    
    # Agreement within 1 point
    metrics['rater1_agreement_1pt'] = np.mean(np.abs(r1_clean - gt_clean) <= 1)
    metrics['rater2_agreement_1pt'] = np.mean(np.abs(r2_clean - gt_clean) <= 1)
    
    # Exact agreement
    metrics['rater1_exact_agreement'] = np.mean(r1_clean == gt_clean)
    metrics['rater2_exact_agreement'] = np.mean(r2_clean == gt_clean)
    
    # Number of valid comparisons
    metrics['n_valid_comparisons'] = len(gt_clean)
    
    return metrics

def process_patient_data(patient_dir, patient_name):
    """
    Process a single patient's retrospective SBS data.
    
    Parameters:
    patient_dir: Path to patient directory
    patient_name: Name of the patient
    
    Returns:
    dict: IRR metrics for this patient
    """
    retro_file = os.path.join(patient_dir, f'{patient_name}_SBS_Scores_Retro.xlsx')
    
    if not os.path.exists(retro_file):
        print(f"No retrospective file found for {patient_name}")
        return None
    
    try:
        # Load the Excel file
        df = pd.read_excel(retro_file)
        
        # Check if required columns exist
        required_cols = ['SBS_1', 'SBS_2', 'SBS']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing columns for {patient_name}: {missing_cols}")
            return None
        
        # Extract scores
        rater1_scores = pd.to_numeric(df['SBS_1'], errors='coerce')
        rater2_scores = pd.to_numeric(df['SBS_2'], errors='coerce')
        ground_truth_scores = pd.to_numeric(df['SBS'], errors='coerce')
        
        # Calculate IRR metrics
        metrics = calculate_irr_metrics(rater1_scores, rater2_scores, ground_truth_scores)
        
        if metrics is None:
            print(f"Insufficient valid data for {patient_name}")
            return None
        
        # Add patient info
        metrics['patient'] = patient_name
        metrics['total_scores'] = len(df)
        
        return metrics
        
    except Exception as e:
        print(f"Error processing {patient_name}: {str(e)}")
        return None

def calculate_population_metrics(all_patient_metrics):
    """
    Calculate population-level IRR metrics.
    
    Parameters:
    all_patient_metrics: List of patient-level metrics dictionaries
    
    Returns:
    dict: Population-level metrics
    """
    if not all_patient_metrics:
        return None
    
    # Combine all valid scores across patients
    all_rater1 = []
    all_rater2 = []
    all_ground_truth = []
    
    for patient_metrics in all_patient_metrics:
        if patient_metrics is None:
            continue
            
        # We need to reload the data to get the actual scores
        patient_name = patient_metrics['patient']
        patient_dir = os.path.join('data_analysis/Misc_analysis/PatientData', patient_name)
        retro_file = os.path.join(patient_dir, f'{patient_name}_SBS_Scores_Retro.xlsx')
        
        try:
            df = pd.read_excel(retro_file)
            rater1_scores = pd.to_numeric(df['SBS_1'], errors='coerce')
            rater2_scores = pd.to_numeric(df['SBS_2'], errors='coerce')
            ground_truth_scores = pd.to_numeric(df['SBS'], errors='coerce')
            
            # Add valid scores to population
            valid_mask = ~(rater1_scores.isna() | rater2_scores.isna() | ground_truth_scores.isna())
            all_rater1.extend(rater1_scores[valid_mask].tolist())
            all_rater2.extend(rater2_scores[valid_mask].tolist())
            all_ground_truth.extend(ground_truth_scores[valid_mask].tolist())
            
        except Exception as e:
            print(f"Error loading data for population analysis: {patient_name} - {str(e)}")
    
    if len(all_ground_truth) < 2:
        print("Insufficient data for population analysis")
        return None
    
    # Calculate population metrics
    population_metrics = calculate_irr_metrics(
        pd.Series(all_rater1), 
        pd.Series(all_rater2), 
        pd.Series(all_ground_truth)
    )
    
    if population_metrics:
        population_metrics['patient'] = 'POPULATION'
        population_metrics['total_scores'] = len(all_ground_truth)
        population_metrics['n_patients'] = len([m for m in all_patient_metrics if m is not None])
    
    return population_metrics

def create_summary_plots(all_patient_metrics, population_metrics, output_dir):
    """
    Create summary plots for IRR analysis.
    
    Parameters:
    all_patient_metrics: List of patient-level metrics
    population_metrics: Population-level metrics
    output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Filter out None values
    valid_metrics = [m for m in all_patient_metrics if m is not None]
    
    if not valid_metrics:
        print("No valid metrics to plot")
        return
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Interrater Reliability Summary', fontsize=16)
    
    # Plot 1: Pearson correlation by patient
    patients = [m['patient'] for m in valid_metrics]
    rater1_pearson = [m['rater1_pearson_r'] for m in valid_metrics]
    rater2_pearson = [m['rater2_pearson_r'] for m in valid_metrics]
    
    x = np.arange(len(patients))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, rater1_pearson, width, label='Rater 1', alpha=0.7)
    axes[0, 0].bar(x + width/2, rater2_pearson, width, label='Rater 2', alpha=0.7)
    axes[0, 0].set_xlabel('Patient')
    axes[0, 0].set_ylabel('Pearson Correlation')
    axes[0, 0].set_title('Pearson Correlation by Patient')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(patients, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MAE by patient
    rater1_mae = [m['rater1_mae'] for m in valid_metrics]
    rater2_mae = [m['rater2_mae'] for m in valid_metrics]
    
    axes[0, 1].bar(x - width/2, rater1_mae, width, label='Rater 1', alpha=0.7)
    axes[0, 1].bar(x + width/2, rater2_mae, width, label='Rater 2', alpha=0.7)
    axes[0, 1].set_xlabel('Patient')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('MAE by Patient')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(patients, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Agreement within 1 point
    rater1_agree = [m['rater1_agreement_1pt'] for m in valid_metrics]
    rater2_agree = [m['rater2_agreement_1pt'] for m in valid_metrics]
    
    axes[1, 0].bar(x - width/2, rater1_agree, width, label='Rater 1', alpha=0.7)
    axes[1, 0].bar(x + width/2, rater2_agree, width, label='Rater 2', alpha=0.7)
    axes[1, 0].set_xlabel('Patient')
    axes[1, 0].set_ylabel('Agreement within 1 point')
    axes[1, 0].set_title('Agreement within 1 point by Patient')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(patients, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Number of valid comparisons
    n_comparisons = [m['n_valid_comparisons'] for m in valid_metrics]
    
    axes[1, 1].bar(patients, n_comparisons, alpha=0.7)
    axes[1, 1].set_xlabel('Patient')
    axes[1, 1].set_ylabel('Number of Valid Comparisons')
    axes[1, 1].set_title('Number of Valid Comparisons by Patient')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'irr_summary_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create population comparison plot if available
    if population_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Population metrics comparison
        metrics_names = ['Pearson R', 'Spearman R', 'MAE', 'RMSE', 'Kappa', 'Agreement 1pt', 'Exact Agreement']
        rater1_pop = [population_metrics['rater1_pearson_r'], population_metrics['rater1_spearman_r'],
                      population_metrics['rater1_mae'], population_metrics['rater1_rmse'],
                      population_metrics['rater1_kappa'], population_metrics['rater1_agreement_1pt'],
                      population_metrics['rater1_exact_agreement']]
        rater2_pop = [population_metrics['rater2_pearson_r'], population_metrics['rater2_spearman_r'],
                      population_metrics['rater2_mae'], population_metrics['rater2_rmse'],
                      population_metrics['rater2_kappa'], population_metrics['rater2_agreement_1pt'],
                      population_metrics['rater2_exact_agreement']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[0].bar(x - width/2, rater1_pop, width, label='Rater 1', alpha=0.7)
        axes[0].bar(x + width/2, rater2_pop, width, label='Rater 2', alpha=0.7)
        axes[0].set_xlabel('Metric')
        axes[0].set_ylabel('Value')
        axes[0].set_title('Population-Level IRR Metrics')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics_names, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add population info
        axes[1].text(0.1, 0.8, f"Total Patients: {population_metrics['n_patients']}", fontsize=12)
        axes[1].text(0.1, 0.7, f"Total Comparisons: {population_metrics['total_scores']}", fontsize=12)
        axes[1].text(0.1, 0.6, f"Valid Comparisons: {population_metrics['n_valid_comparisons']}", fontsize=12)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_title('Population Summary')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'population_irr_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Main function to calculate IRR scores for all patients.
    """
    # Set up paths
    patient_data_dir = 'data_analysis/Misc_analysis/PatientData'
    output_dir = 'data_analysis/Misc_analysis/irr_results'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of patient directories
    patient_dirs = [d for d in os.listdir(patient_data_dir) 
                   if os.path.isdir(os.path.join(patient_data_dir, d)) and 'Patient' in d]
    patient_dirs.sort()
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Process each patient
    all_patient_metrics = []
    
    for patient_dir in patient_dirs:
        patient_path = os.path.join(patient_data_dir, patient_dir)
        print(f"\nProcessing {patient_dir}...")
        
        metrics = process_patient_data(patient_path, patient_dir)
        all_patient_metrics.append(metrics)
        
        if metrics:
            print(f"  Valid comparisons: {metrics['n_valid_comparisons']}")
            print(f"  Rater 1 Pearson r: {metrics['rater1_pearson_r']:.3f}")
            print(f"  Rater 2 Pearson r: {metrics['rater2_pearson_r']:.3f}")
            print(f"  Rater 1 Cohen's Kappa: {metrics['rater1_kappa']:.3f}")
            print(f"  Rater 2 Cohen's Kappa: {metrics['rater2_kappa']:.3f}")
            print(f"  Rater 1 MAE: {metrics['rater1_mae']:.3f}")
            print(f"  Rater 2 MAE: {metrics['rater2_mae']:.3f}")
            print(f"  Rater 1 Agreement within 1pt: {metrics['rater1_agreement_1pt']:.1%}")
            print(f"  Rater 2 Agreement within 1pt: {metrics['rater2_agreement_1pt']:.1%}")
        else:
            print(f"  No valid data found")
    
    # Calculate population metrics
    print(f"\nCalculating population-level metrics...")
    population_metrics = calculate_population_metrics(all_patient_metrics)
    
    if population_metrics:
        print(f"Population-level results:")
        print(f"  Total patients: {population_metrics['n_patients']}")
        print(f"  Total comparisons: {population_metrics['total_scores']}")
        print(f"  Valid comparisons: {population_metrics['n_valid_comparisons']}")
        print(f"  Rater 1 Pearson r: {population_metrics['rater1_pearson_r']:.3f}")
        print(f"  Rater 2 Pearson r: {population_metrics['rater2_pearson_r']:.3f}")
        print(f"  Rater 1 Cohen's Kappa: {population_metrics['rater1_kappa']:.3f}")
        print(f"  Rater 2 Cohen's Kappa: {population_metrics['rater2_kappa']:.3f}")
        print(f"  Rater 1 MAE: {population_metrics['rater1_mae']:.3f}")
        print(f"  Rater 2 MAE: {population_metrics['rater2_mae']:.3f}")
        print(f"  Rater 1 Agreement within 1pt: {population_metrics['rater1_agreement_1pt']:.1%}")
        print(f"  Rater 2 Agreement within 1pt: {population_metrics['rater2_agreement_1pt']:.1%}")
    
    # Create summary plots
    print(f"\nCreating summary plots...")
    create_summary_plots(all_patient_metrics, population_metrics, output_dir)
    
    # Save results to CSV
    print(f"\nSaving results...")
    
    # Patient-level results
    valid_patient_metrics = [m for m in all_patient_metrics if m is not None]
    if valid_patient_metrics:
        patient_df = pd.DataFrame(valid_patient_metrics)
        patient_df.to_csv(os.path.join(output_dir, 'patient_irr_metrics.csv'), index=False)
        print(f"Patient-level results saved to: {os.path.join(output_dir, 'patient_irr_metrics.csv')}")
    
    # Population-level results
    if population_metrics:
        population_df = pd.DataFrame([population_metrics])
        population_df.to_csv(os.path.join(output_dir, 'population_irr_metrics.csv'), index=False)
        print(f"Population-level results saved to: {os.path.join(output_dir, 'population_irr_metrics.csv')}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("Summary plots:")
    print(f"  - irr_summary_plots.png")
    if population_metrics:
        print(f"  - population_irr_metrics.png")

if __name__ == "__main__":
    main()