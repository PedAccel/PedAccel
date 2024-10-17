#Import Necessary Libraries
import General_Functions
import Smoothing_Functions
import Frequency_Domain
import numpy as np
from pygt3x.reader import FileReader
from matplotlib import pyplot as plt
import Actigraph_Metrics
import Correlation_and_Analysis
import pandas as pd
import skdh
import scipy
import random
from scipy.fftpack import fft, fftfreq

from Modules import General_Functions, Frequency_Domain


print("This file __name__ is set to: {}" .format(__name__))

with FileReader("No Movement.gt3x") as reader:
      myData = reader.to_pandas()
print(myData.head(5))


General_Functions.create_time_column(myData)
General_Functions.create_absMag_Column(myData)

#Slice Data because of movement at end of file
myData = myData.iloc[:32000]
signal = myData['VecMag']

#Generate FFT Data
freqData, freqBins = Frequency_Domain.FFTData(signal,100)
#Plot FFT Data
Frequency_Domain.FFTPlot(freqData,freqBins, 100, lower_freq=.3, upper_freq=20, lower_cutoff=.3, upper_cutoff=100000,title='FFT of raw Data')

#My comb filter
Frequency_Domain.myCombFilter(signal)

#scipy comb filter
Frequency_Domain.scipy_comb_filter(signal)

#Rolling average in Frequency Domain
Frequency_Domain.freq_domain_rollingAvg(signal,wlen=11)

#Average out harmonics every f0Hz in frequency domain
freqData, freqBins = Frequency_Domain.average_out_harmonics(signal,f0=1,fmax=20)
print(freqData)
#Inverse FFT
N = len(freqData)
itx = Frequency_Domain.inverse_FFT(freqData)
plt.scatter(myData['Time(s)'], itx)
plt.title('IFFT after average out harmonics in freq domain')
plt.show()

#Average out outliers in frequency domain
freqData,freqBins = Frequency_Domain.remove_outliers_freqDomain(signal,wlen=20)
#Inverse FFT on averaged outliers
N = len(freqData)
itx = Frequency_Domain.inverse_FFT(freqData)
print(itx)
print(len(myData['Time(s)']))
plt.scatter(myData['Time(s)'], itx)
plt.title('IFFT after removed outliers in freq domain')
plt.show()
