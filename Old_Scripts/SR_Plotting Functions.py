import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.signal import cwt, morlet
from matplotlib.colors import Normalize

'''
Sid's Testing Functions:
Functions and Test Code Included
'''

def create_time_column(myData):
    """
    This Function Creates a time column in the Accelerometry Data Frame
    :param myData: gt3x file containing accelerometry data
    :return: Updated Data Frame with Time Column
    """
    frequency = 100  # Hertz
    time_Column_Array = []
    # length of time column is the same as the length of a data column
    length_of_Data = len(myData["X"])
    for i in range(1, length_of_Data + 1):
        time_Column_Array.append(i * (1 / frequency))
    myData['Time(s)'] = time_Column_Array
    print(myData.head(5))

def create_absMag_Column(myData):
    VecMag = np.sqrt(((myData["X"]) ** 2) + ((myData["Y"]) ** 2) + ((myData["Z"]) ** 2))
    myData['VecMag'] = VecMag

def plot_Data(myData, xData_name, yData_name, title='Raw Data'):
    # Data to work with
    y = myData[yData_name]
    x = myData[xData_name]

    # Plotting
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xData_name)
    plt.ylabel(yData_name)
    plt.grid(True)
    plt.show()

def FFT(y, sr=100, lower_freq=0, upper_freq=20, lower_cutoff=40, upper_cutoff=10000, title='FFT Plot', figsize=(12, 6), xlim=None, ylim=None):
    """
    :param myData: gt3x file containing accelerometry data
    :param yData_name: Column to compute the Fourier Transfrom of
    :param sr: Sampling rate, default to 100 Hz
    :param lower_freq: Minimum Frequency to Plot, default to 0Hz
    :param upper_freq: Maximum frequency to Plot, default to 20Hz
    :param lower_cutoff: Minimum Magnitude to Plot, default to 1000
    :param upper_cutoff: Maximum Magnitude to plot, default to 20000
    :param figsize: Size of the figure, default to (12, 6)
    :param xlim: Tuple specifying the limits of the x-axis, default to None
    :param ylim: Tuple specifying the limits of the y-axis, default to None
    :return: Plot of frequency and magnitudes
    """
    input_data = np.array(y)
    X = fft(input_data)
    N = len(X)
    n = np.arange(N)
    new_array = np.abs(X)
    #create an array with the proper frequencies
    T = N / sr
    freq = n / T
    upper_freq_index = int(upper_freq*T)
    lower_freq_index = int(lower_freq*T)
    # Band pass filter with lower cutoff and upper cutoff
    new_array[new_array > upper_cutoff] = 0
    new_array[new_array < lower_cutoff] = 0
    # Get the one-sided spectrum
    n_oneside = N // 2
    # get the one side frequency
    f_oneside = freq[:n_oneside]
    
    # Calculate power spectral density (PSD)
    psd = (1 / (sr * N)) * np.square(new_array)
    
    # Calculate the area under the curve
    area = np.trapz(new_array[lower_freq_index:upper_freq_index], x=f_oneside[lower_freq_index:upper_freq_index])
    print("Area under the curve:", area)
    
    # Print PSD
    print("Power Spectral Density (PSD):", psd[lower_freq_index:upper_freq_index].sum())
    
    #Plot frequency decomposition
    plt.figure(figsize=figsize)
    print(f'The maximum frequency is {(max( (v, i) for i, v in enumerate(new_array[lower_freq_index:upper_freq_index]) )[1])/T}')
    plt.plot(f_oneside[lower_freq_index:upper_freq_index], new_array[lower_freq_index:upper_freq_index])
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.show()

def butter_bandpass_filter(data, lower_cutoff, upper_cutoff, fs=100, order=4):
    nyquist = 0.5 * fs
    low = lower_cutoff / nyquist
    high = upper_cutoff / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered_data = scipy.signal.filtfilt(b, a, data)
    return filtered_data

def FFTData(signal, sr):
    n = len(signal)
    freqData = np.fft.fft(signal) / n
    freqData = 2 * np.abs(freqData[:n//2])
    freqBins = np.fft.fftfreq(n, 1/sr)[:n//2]
    return freqData, freqBins

def FFTPlot(freqData, freqBins, sr, lower_freq, upper_freq, lower_cutoff, upper_cutoff, title):
    plt.figure(figsize=(10, 6))
    plt.plot(freqBins, freqData, color='blue')
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.xlim([lower_freq, upper_freq])
    plt.ylim([lower_cutoff, upper_cutoff])
    plt.grid()
    plt.show()

def scipy_comb_filter(signalcf, fs=100, f0=1, Q=20):  # Adjusted Q value
    # Adjust bandpass filter parameters based on your signal characteristics
    #signalcf = butter_bandpass_filter(signalcf, lower_cutoff=0.3, upper_cutoff=7)

    # Design comb filter with adjusted parameters
    b, a = scipy.signal.iircomb(f0, Q, ftype='notch', fs=fs)

    # Normalize filter coefficients
    b /= a[0]
    a /= a[0]

    # Check the frequency response of the comb filter
    w, h = scipy.signal.freqz(b, a)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.title('Comb Filter Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.grid()
    plt.show()

    # Apply the comb filter
    y_filter = scipy.signal.lfilter(b, a, signalcf)

    # Plot the FFT of the filtered signal
    freqData, freqBins = FFTData(y_filter, fs)
    FFTPlot(freqData, freqBins, sr=fs, lower_freq=0.3, upper_freq=20, lower_cutoff=0, upper_cutoff=40, title='FFT with scipy comb filter')

    # Return filtered time series data
    return y_filter

def plot_continuous_wavelet_transform(signal, t, widths=None, wavelet=morlet, figsize=(10, 6)):
    """
    Perform and plot the continuous wavelet transform of a time series signal.

    Parameters:
    - signal: The time series signal.
    - t: Time values corresponding to the signal.
    - widths: Range of scales (widths) for the wavelet. Default is None.
    - wavelet: The wavelet function to be used. Default is morlet.
    - figsize: Size of the figure. Default is (10, 6).
    """

    if widths is None:
        widths = np.arange(1, 500)

    # Perform continuous wavelet transform
    cwt_result = cwt(signal, wavelet, widths)

    # Normalize the CWT result for better visualization
    norm = Normalize(vmin=np.min(np.abs(cwt_result)), vmax=np.max(np.abs(cwt_result)))

    # Plot the original signal
    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Original Time Series Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Plot the CWT result
    plt.subplot(2, 1, 2)
    plt.imshow(np.abs(cwt_result), extent=[np.min(t), np.max(t), np.min(widths), np.max(widths)],
               cmap='jet', aspect='auto', interpolation='bilinear', norm=norm)
    plt.colorbar(label='Magnitude')
    plt.title('Continuous Wavelet Transform (CWT)')
    plt.xlabel('Time')
    plt.ylabel('Scale (Width of Wavelet)')

    plt.tight_layout()
    plt.show()

def calculate_psd(time, acceleration):
    # Calculate the Power Spectral Density (PSD) using Fourier Transform
    fs = 1 / (time[1] - time[0])  # Sampling frequency
    f, psd = plt.psd(acceleration, NFFT=len(time), Fs=fs, scale_by_freq=False)

    return f, psd

def plot_psd(frequencies, psd_values):
    # Plot the Power Spectral Density
    plt.figure(figsize=(10, 6))
    plt.loglog(frequencies, psd_values, color='b')
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (g^2/Hz)')
    plt.grid(True)
    plt.show()
    
def moving_average(signal, w):
    """
    :param signala:signal to work with
    :param w: Window of rolling average
    :return: new  signal with rolling average included
    """
    averaged = np.convolve(signal, np.ones(w), 'valid') / w
    return averaged

def median_filter(signal,num_passes, wlen = 301):
    """
    :param signal: signal to modify with Spline Fit Data
    :param num_passes: number of times to smooth the data
    :param wlen: size of window for rolling median
    :return: Median Smoothed signal
    """
    for i in range(num_passes):
        myDataArray = scipy.signal.medfilt(signal, kernel_size=wlen)
    return myDataArray

if __name__ == "__main__":
    sliced_data = pd.read_csv(r"C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\Data Analysis\PythonPipeline\Patient_9_Intervals\SBS1_num1.csv")
    print(sliced_data.head(5))

    # Create New Columns
    create_time_column(sliced_data)
    create_absMag_Column(sliced_data)

    x = sliced_data['Time(s)']
    y = sliced_data['VecMag']

    # Sampling Rate
    sr = 100

    # Plotting data
    # plot_Data(sliced_data, 'Time(s)', 'VecMag', title='Vector Magnitude vs Time')
    
    # Set the limits for x and y-axes
    xlim = (0, 20)
    ylim = (0, 20000)
    med_y = moving_average(y, 51)
    FFT(med_y, sr, lower_freq = 0.3, upper_freq = 30, lower_cutoff = 0, upper_cutoff = 40, title = 'SBS -1: FFT of Raw Data', figsize = (10,6), xlim = (0, 20), ylim = (0, 60))

    # FFT(y, sr, lower_freq =.3, upper_freq=30, lower_cutoff=0, upper_cutoff=40, title='SBS -1: FFT of Raw Data')

    # Comb Filters
    # scipy_comb_filter(y)
    # Frequency_Domain.scipy_comb_filter(y)

    # # Rolling Average Filter
    # Frequency_Domain.freq_domain_rollingAvg(y, wlen=11)

    # # Power Spectral Density
    # frequencies, psd_values = calculate_psd(x, y)
    # plot_psd(frequencies, psd_values)

    # # Wavelet Transform
    # plot_continuous_wavelet_transform(y, x)
    

    # folder_path = 'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\Data Analysis\PythonPipeline\Patient_9_5MIN1SW'

    # # List all CSV files in the folder
    # csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # # Iterate through each CSV file
    # for file in csv_files:
    #     # Read the CSV file
    #     df = pd.read_csv(os.path.join(folder_path, file))
    
    # # Assuming the column containing the data is named 'data_column'
    # # You need to replace 'data_column' with the actual name of your data column
    # data_column = df['data_column'].values
    
    # # Run FFT on the data
    # FFT(data_column, title=file)
