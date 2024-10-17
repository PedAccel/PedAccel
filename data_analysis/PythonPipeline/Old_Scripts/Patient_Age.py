import pandas as pd
import matplotlib.pyplot as plt
import os

data_dir = './PatientData/Patient9'
filename = "Patient_Age.xlsx"  # Update with the path to your Excel file
fp = os.path.join(data_dir, filename)

df = pd.read_excel(filename)
# Extract ages from a specific column (assuming column name is 'Age')
ages = df['Age (months)']

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(ages, bins=20, color='skyblue', edgecolor='black')  # Adjust number of bins as needed
plt.xlabel('Age (Months)')
plt.ylabel('Count')
plt.title('Distribution of Patient Ages')
plt.grid(False)
plt.show()