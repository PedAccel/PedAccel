import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_df(file_path):
    df_ecg = pd.read_csv(file_path)
    return df_ecg

def make_boxplot(df, pat_num=None, output_path=None, save=False):
    features = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_LFHF']
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, feature in zip(axes, features):
        df.boxplot(column=feature, by='SBS_SCORE', grid=False, ax=ax)
        if pat_num:
            ax.set_title(f'Patient {pat_num} : {feature} by SBS_SCORE')
        else:
            ax.set_title(f'{feature} by SBS_SCORE')
        ax.set_xlabel('SBS_SCORE')
        ax.set_ylabel('Value')

    plt.suptitle('')
    plt.tight_layout()

    if save and output_path:
        plt.savefig(output_path)
        print(f'Saved box plot at {output_path}')

    plt.show()

def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    pat_nums = [4, 5, 6, 8, 9, 13]

    for i in tqdm(pat_nums):
        pat_num = 'pat' + str(i) + '_'
        output_path = os.path.join(output_dir, pat_num + 'boxplot.png')

        df_ecg = load_df(os.path.join(output_dir, pat_num + 'df_ecg1.csv'))

        df = pd.DataFrame({
            'SBS_SCORE': df_ecg['SBS_SCORE'],
            'HRV_MeanNN': df_ecg['HRV_MeanNN'],
            'HRV_SDNN': df_ecg['HRV_SDNN'],
            'HRV_LFHF': df_ecg['HRV_LFHF'],
            })
        
        make_boxplot(df, i, output_path, save=True)

        print(f'Generated box plot for Patient {i}')

if __name__ == "__main__":
    main()
