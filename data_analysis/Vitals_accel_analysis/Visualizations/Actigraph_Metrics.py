#This Script is used to generate actigraphy features and metrics for Analysis 

# add better documentation here

#Load Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
# import skdh #Scikit-Digital-Health for pip install
import pandas as pd
import tsfel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.lines as mlines
from scipy.io import loadmat
from scipy.stats import pearsonr
import os

#Calculates area under the curve of some input signal
def calc_area_under_curve(x,y):
    """
        :param x: x data
        :param y: y data
        :return: Area under the curve (an integration of the data)
    """
    return np.trapz(y, x, dx=.005, axis=-1)


#Scipy Peak Finder Function
def Scipy_find_peaks(x):
    """
        :param x: data to find peaks of
        :return: Plot of peaks generated, number of peaks returned
    """
    #Prominence is most important parameter, but others should be explored
    peaks, _ = scipy.signal.find_peaks(x, prominence=1) 
    plt.plot(peaks, x[peaks], "ob")
    print(f'number of peaks for prominence parameter is {len(peaks)}')
    plt.plot(x)
    plt.show()
    return len(peaks)

#Generate MAD data...To Do: Double check this with the scipy version
def VecMag_MAD(signal, window=100):
    """
    Computes the Mean Absolute Deviation (MAD) of the vector magnitude data. The signal is chucked into windows (non-overlapping) and MAD is computed for each window.

    :param signal: Vector Magnitude Data to compute MAD for
    :param wlen: Length of window to generate a MAD data point for
    :return: list of int. of length signal/wlen + 1
    """
    MAD = []
    start_time = np.arange(0, len(signal), window)
    for i in start_time[:-1]:
        signal_window = signal[i:i + window]
        MAD.append(scipy.stats.median_abs_deviation(signal_window, axis=0, nan_policy='propagate'))

    # last window
    signal_window = signal[start_time[-1]:]
    MAD.append(scipy.stats.median_abs_deviation(signal_window, axis=0, nan_policy='propagate'))

    return MAD

##Generated MAD data from triaxial raw data 
# skdh seems to be in active development and there's no documentations, let's move away for now
# def skdh_MAD(sliced_data,wlen = 100):
#     shape = (len(sliced_data['X']), 3)
#     arr = np.ones(shape)
#     arr[0:,0] = sliced_data['X']
#     arr[0:, 1] = sliced_data['Y']
#     arr[0:, 2] = sliced_data['Z']
#     accel = arr
#     MAD = skdh.activity.metric_mad(accel, wlen)
#     myX = []
#     for i in range(len(MAD)):
#         myX.append(i)
#     return myX, MAD

#Calculates simple signal to noise ratio
def Signal_To_Noise_Ratio(signal):
    std = np.std(signal)
    mean = np.mean(signal)
    return mean/std

def Feature_Extraction(location, filename, fs=100):
    filepath = os.path.join(location, filename)
    data = loadmat(filepath)
    x_mag = data["x_mag"]
    sbs = data["sbs"]

    # Generate configuration file for feature extraction
    cfg_file = tsfel.get_features_by_domain()
    print(x_mag.shape)

    # Extract features and restructure data
    features_list = []
    sbs_list = []
    for i in range(x_mag.shape[0]):
        features = tsfel.time_series_features_extractor(cfg_file, x_mag[i, :], fs=fs, verbose=0)
        features_list.append(features)
        # FIXME: why are we only taking the first element of sbs?
        sbs_list.append(sbs[0][i])
    
    # Convert features and SBS scores to DataFrame
    # where did 389 cpme from?
    features_array = np.array(features_list).reshape(-1, 389)


    # why are are making a dataframe that we are not using?
    df_features = pd.DataFrame(features_array)
    df_features.columns = ['feature_' + str(col) for col in df_features.columns]

    df_sbs = pd.DataFrame({'SBS': sbs_list})
    
    df = pd.concat([df_sbs, df_features], axis=1)
    df.head(10)
    x = df_features  # Features DataFrame
    y = df['SBS'] 
    
    return x, y

