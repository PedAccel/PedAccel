# Import Modules
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from openpyxl import load_workbook
from scipy.io import savemat, loadmat
import os

# Set directory
directory = os.chdir(r'S:\Sedation_monitoring\Sickbay_extract\Ventilator_Data\Study57_Tag123_EventList')

# File path
file = r'S:\Sedation_monitoring\sickbay_extract\Ventilator_Data\Study57_Tag123_EventList\Event_Row_11_Data_zero_order_interpolation.csv'

# Create directory for images if it doesn't exist
image_dir = 'plots'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Load the CSV file
df = pd.read_csv(file)

# Get the relative time column (first column)
relative_time = df.iloc[:, 0]

# Iterate through all columns except 'Relative Time' and 'Time'
for column in df.columns:
    if column not in ['Relative Time', 'Time']:
        # Check if the column has any non-null data
        if df[column].notna().any():
            # Create a new figure for each plot
            plt.figure(figsize=(10, 6))
            
            # Plot the data
            plt.plot(relative_time, df[column], linewidth=1)
            
            # Set labels and title
            plt.xlabel('Relative Time (seconds)', fontsize=12)
            plt.ylabel(column, fontsize=12)
            plt.title(f'{column} vs Relative Time', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Save the figure
            # Clean the column name for filename (remove special characters)
            safe_filename = "".join(c for c in column if c.isalnum() or c in (' ', '_', '-')).rstrip()
            safe_filename = safe_filename.replace(' ', '_')
            filepath = os.path.join(image_dir, f'{safe_filename}.png')
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f'Saved plot for {column}')

print(f'\nAll plots saved to: {os.path.join(directory, image_dir)}')