#Import Necessary Libraries
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import splrep, BSpline
import pywt #installed package name is pywavelets
import math


def moving_average(signal, w=50):
    """
    :param signala:signal to work with
    :param w: Window of rolling average
    :return: new  signal with rolling average included
    """
    averaged = np.convolve(signal, np.ones(w), 'valid') / w
    return averaged

def median_filter(signal,num_passes, wlen = 51):
    """
    :param signal: signal to modify with Spline Fit Data
    :param num_passes: number of times to smooth the data
    :param wlen: size of window for rolling median
    :return: Median Smoothed signal
    """
    for i in range(num_passes):
        myDataArray = scipy.signal.medfilt(signal, kernel_size=wlen)
    return myDataArray

def down_sample_to50Hz(signal):
    """
      :param signal: signal to work with
      :return: new signal with 50Hz sampling 
      """
    y1 = np.array(signal)
    avgy = ((y1 + np.roll(y1, 1)) / 2.0)[1::2]
    return avgy

def Spline_Regression(x,signal,smooth_param,title='Spline Fit Curve'):
    """
    :param signal: signal to modify with Spline Fit Data
    :param x: X data
    :param y: Y data
    :param smooth_param: Larger smooth_param means more smoothing of the raw data
    :return: Generates a plot of smoothed Data
    """
    tck = splrep(x, signal, s=smooth_param)
    plt.plot(x, BSpline(*tck)(x))
    plt.title(title)
    plt.show()


#Wavelet Smoothing Function
def Wavelet_Smoothing(x,y, threshold=.5):
    signal = y
    coeffs = pywt.wavedec(signal, 'db1', level=6)
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    # Reconstruct the signal from the thresholded coefficients
    denoised_signal = pywt.waverec(coeffs_thresholded, 'db1')
    # Plotting the noisy and denoised signals
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, signal)
    plt.title("Noisy Signal")
    plt.subplot(1, 2, 2)
    plt.plot(x, denoised_signal)
    plt.title("Denoised Signal")
    plt.tight_layout()
    plt.show()

def threshold_data(signal, cutoff):
    signal = np.array(signal)
    low = signal < cutoff
    signal[low] = 0
    return signal


def ENMO(signal,threshold = True):
    signal = np.array(signal)
    signal = abs(signal) - 1
    if(threshold):
        signal = threshold_data(signal, cutoff = .03)
    return signal
