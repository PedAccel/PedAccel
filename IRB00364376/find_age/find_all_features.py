#%%

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow as pa
import seaborn as sns
from datetime import timedelta, datetime
import torch
import pandas as pd

dir = Path(r"S:\Fackler_OSS_364376\data\IRB-364376-v1-230215")
# %%
import os
ehr_dir = dir.joinpath('EHR')
file_num_f = open('file_num_features.csv', 'w')
nan_percentage = open('nan_percentage.csv', 'w')
file_num_f.write('name,num_entries,\n')
nan_percentage.write('feature, nan_percentage, file,\n')

for file in os.listdir(ehr_dir):
    if file.endswith('.csv.gz'):
        fp = dir.joinpath('EHR', file)
        data = pd.read_csv(fp, compression="gzip")
        file_num_f.write(f'{file}, {str(data.shape[0])}, \n')
        
        for feature in data.columns.values:
            nan_percentage.write(f'{feature}, {str(data[feature].isna().sum()/len(data[feature]))}, {file}, \n' )
            
# %%
