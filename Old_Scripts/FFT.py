#This workbook is used to filter and generate FFT Plots using Frequency_Domain library

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os

# Load Data
data_dir = './PatientData/Patient9'
filename = 'pt9_win5_5.mat'

fp = os.path.join(data_dir, filename)
data = loadmat(fp)

total_signal = data["x_mag"]
#Use a specifc window from the signal array
signal = total_signal[0] #Change the index to generate different FFTs
sbs = data["sbs"]

sampling_rate = 100
signal = signal[::sampling_rate]
# TODO: maybe median is better




# #Generate FFT Data
# freqData, freqBins = Frequency_Domain.FFTData(signal,100)
# #Plot FFT Data
# Frequency_Domain.FFTPlot(freqData,freqBins, 100, lower_freq=.1, upper_freq=20, lower_cutoff=.3, upper_cutoff=100000,title='FFT')

# #scipy comb filter
# Frequency_Domain.scipy_comb_filter(signal)

# #Average out harmonics every f0Hz in frequency domain
# #if f0 is not a frequency bin, code will give run time error
# freqData, freqBins = Frequency_Domain.average_out_harmonics(signal,f0=1,fmax=20)
# #Inverse FFT after averaging out harmonics
# itx = Frequency_Domain.inverse_FFT(freqData)
# plt.scatter(time, itx)
# plt.title('IFFT after average out harmonics in freq domain')
# plt.show()

# #Average out outliers in frequency domain
# freqData,freqBins = Frequency_Domain.remove_outliers_freqDomain(signal,wlen=5)
# #Inverse FFT on averaged outliers
# itx = Frequency_Domain.inverse_FFT(freqData)
# plt.scatter(time, itx)
# plt.title('IFFT after removed outliers in freq domain')
# plt.show()
