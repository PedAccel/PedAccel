import os
from tqdm import tqdm
import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def make_boxplot(df, pat_num=None, ecg_num=None, save=False, show=False):
    features = df.columns.drop('SBS_SCORE')

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    for feature in features:
        scores = sorted(df['SBS_SCORE'].unique())
        data = [df[df['SBS_SCORE'] == score][feature] for score in scores]
        plt.boxplot(data, labels=scores)
        plt.title(f'Patient {pat_num} ECG {ecg_num} {feature}')
        plt.xlabel('SBS Score')
        plt.ylabel(feature)

        if save:
            plt.savefig(os.path.join(os.path.dirname(__file__), 'output', f'Patient{pat_num}_ECG{ecg_num}_{feature}.png'))

        if show:
            plt.show()

        plt.close()
    
    return None

def correlation_matrix(df, pat_num=None, ecg_num=None, save=False, show=False):
    corr = df.corr()

    if save:
        corr.to_csv(os.path.join(os.path.dirname(__file__), 'output', f'Patient{pat_num}_ECG{ecg_num}_correlation_matrix.csv'))

    if show:
        correlation_plot(df, pat_num, ecg_num, False, True)

    return None

def correlation_plot(df, pat_num=None, ecg_num=None, save=False, show=False):
    corr = df.corr()

    plt.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()

    if save:
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output', f'Patient{pat_num}_ECG{ecg_num}_correlation_matrix.png'))

    if show:
        plt.show()

    plt.close()

    return None

def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        print('Data directory does not exist')
        return

    pat_nums = [4, 9]

    for i in tqdm(pat_nums):
        print(f'Patient {i}')

        _, df_ecg2, _ = preprocessing.load_dfs(output_dir, data_dir, i)

        df_ecg2 = df_ecg2[(df_ecg2['SBS_SCORE'] == -1) | (df_ecg2['SBS_SCORE'] == 2)]
        df_ecg2= df_ecg2.dropna(axis=1, how='any')

        make_boxplot(df_ecg2, i, 2, False, False)
        correlation_matrix(df_ecg2, i, 2, False, False)
        correlation_plot(df_ecg2, i, 2, False, False)

if __name__ == "__main__":
    main()