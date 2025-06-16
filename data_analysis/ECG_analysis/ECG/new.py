import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import tsfel

import preprocessing

def make_boxplot(df, pat_num=None, ecg_num=None, save=False, show=False):
    features = df.columns.drop('TM SBS')

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    for feature in features:
        scores = sorted(df['TM SBS'].unique())
        data = [df[df['TM SBS'] == score][feature] for score in scores]
        plt.boxplot(data, labels=scores)
        plt.title(f'Patient {pat_num} ECG {ecg_num} {feature} TM')
        plt.xlabel('TM SBS')
        plt.ylabel(feature)

        if save:
            plt.savefig(os.path.join(os.path.dirname(__file__), 'output', f'Patient{pat_num}_ECG{ecg_num}_{feature}_TM.png'))

        if show:
            plt.show()

        plt.close()
    
    return None

def correlation_matrix(df, pat_num=None, ecg_num=None, save=False, show=False):
    corr = df.corr()

    if save:
        corr.to_csv(os.path.join(os.path.dirname(__file__), 'output', f'Patient{pat_num}_ECG{ecg_num}_correlation_matrix_TM.csv'))

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
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output', f'Patient{pat_num}_ECG{ecg_num}_correlation_matrix_TM.png'))

    if show:
        plt.show()

    plt.close()

    return None

def standardize_data(df_ecg):
    if df_ecg.isna().any().any():
        df_ecg.dropna(inplace=True)

    for key in df_ecg.keys():
        df_ecg[key] = StandardScaler().fit_transform(df_ecg[key].values.reshape(-1, 1))
    
    return df_ecg

def split_data(df_ecg):
    df_ecg = df_ecg.dropna(axis='columns')

    X = df_ecg.drop(columns=['TM SBS'])

    y = df_ecg['TM SBS']
    X = df_ecg.drop(columns=['TM SBS'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_train = standardize_data(X_train)
    X_test = standardize_data(X_test)

    return X_train, X_test, y_train, y_test

def get_pca_ecg(df_ecg, pat_num=None, ecg_num=None, n_components=2, save=False):
    X_train, X_test, y_train, y_test = split_data(df_ecg)
    
    X = pd.concat([X_train, X_test])

    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    if save and pat_num is not None and ecg_num is not None:
        df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        df['SBS_SCORE'] = pd.concat([y_train, y_test]).values
        output_path = os.path.join(os.path.dirname(__file__), 'output', f'pat{pat_num}_ecg{ecg_num}_pca_tm.csv')
        df.to_csv(output_path, index=False)
        print(f'Saved PCA data at {output_path}')
        
    return pca

def plot_pca_ecg(df_ecg, pat_num=None, ecg_num=None, n_components=None, save=False, show=False):
    X_train, X_test, y_train, y_test = split_data(df_ecg)

    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    color_map = {-3: 'purple', -2: 'blue', -1: 'green', 0: 'yellow', 1: 'orange', 2: 'red'}
    for category, color in color_map.items():
        indices = y == category
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], color=color, label=str(category), edgecolor='k', s=80)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Principal Component Analysis TM SBS')
    plt.legend(title='TM SBS')
    plt.grid()

    if save and pat_num is not None and ecg_num is not None:
        output_path = os.path.join(os.path.dirname(__file__), 'output', f'pat{pat_num}_ecg{ecg_num}_pca_plot_tm.png')
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output', output_path))
        print(f'Saved PCA plot at {output_path}')

    if show:
        plt.show()

def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        print('Data directory does not exist')
        return

    pat_nums = [4]

    for i in tqdm(pat_nums):
        retro_df = preprocessing.load_retro(data_dir, i)
        sickbay_df = preprocessing.load_sickbay(data_dir, i)

        retro_df = retro_df[['Time', 'TM SBS']]
        print(retro_df.shape)
        
        # ecg1_df, ecg2_df, ecg3_df = preprocessing.load_dfs(output_dir, data_dir, i)

        # ecg1_df = preprocessing.match_times(retro_df, ecg1_df)
        # ecg1_df = ecg1_df.drop(['Time', 'start_time', 'end_time', 'SBS_SCORE'], axis=1)

        # ecg2_df = preprocessing.match_times(retro_df, ecg2_df)
        # ecg2_df = ecg2_df.drop(['Time', 'start_time', 'end_time', 'SBS_SCORE'], axis=1)

        # ecg3_df = preprocessing.match_times(retro_df, ecg3_df)
        # ecg3_df = ecg3_df.drop(['Time', 'start_time', 'end_time', 'SBS_SCORE'], axis=1)

        # make_boxplot(ecg1_df, i, 1, True, False)
        # correlation_matrix(ecg1_df, i, 1, True, False)
        # correlation_plot(ecg1_df, i, 1, True, False)

        # get_pca_ecg(ecg1_df, i, 1, 2, True)
        # plot_pca_ecg(ecg1_df, i, 1, 2, True, False)

        # get_pca_ecg(ecg2_df, i, 2, 2, True)
        # plot_pca_ecg(ecg2_df, i, 2, 2, True, False)

        # get_pca_ecg(ecg3_df, i, 3, 2, True)
        # plot_pca_ecg(ecg3_df, i, 3, 2, True, False)

        sickbay_df = preprocessing.match_times(retro_df, sickbay_df)
        sickbay_df = sickbay_df.drop(['Time', 'start_time', 'end_time', 'SedPRN', 'sbs'], axis=1)

        print(sickbay_df.shape)
        print(sickbay_df)

        features = []

        cfg = tsfel.get_features_by_domain()
        
        for j in tqdm(range(len(sickbay_df))):
            data = sickbay_df.iloc[j, :-1].values
            f = tsfel.time_series_features_extractor(cfg, data)
            features.append(f.T)
        
        features_df = pd.concat(features, axis=0).reset_index(drop=True)
        features_df['TM SBS'] = sickbay_df['TM SBS'].values

        features_df.to_csv(os.path.join(output_dir, f'pat{i}_sickbay_tm.csv'), index=False)

if __name__ == "__main__":
    main()
