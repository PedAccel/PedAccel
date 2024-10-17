'''
Plotting functions for raw accelerometry and vitals data.

To be called in the respective python notebooks.
'''

# Import Packages
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import Actigraph_Metrics
# import datetime

def raw_accel_plot(sbs, signal, window_size, plot):
    '''
    Raw Accelerometry Data Plotting Function
    '''
    
def mad_plot(sbs, x_mag, window_size, figure=None):
    '''
    Mean Amplitude Deviation Plotting Function
    @param sbs: Raw SBS Scores
    @param x_mag: Raw Accelerometry Data
    @param window_size: MAD Window Size
    @param figure: 
    '''
    # If figure is not provided, create a new figure
    if figure is None:
        figure = plt.figure(figsize=(8, 6))
    
    # Accelerometry Sampling Frequency
    freq = 100
    
    count = 0
    
    # Print Graph per SBS
    for i, sbs_value in enumerate(sbs): 
        signal = x_mag[i,:]
        signal_mad = Actigraph_Metrics.VecMag_MAD(signal, window_size)
        print(len(signal_mad))

        t = np.arange(0, len(signal), step=window_size) / (freq * 60)
        
        # Plot MAD against time
        plt.figure(figure.number)
        plt.plot(t, signal_mad, color='blue')
        # SBS marker
        plt.axvline(t[len(t)//2], color='red', linestyle='--')
        plt.text(t[len(t)//2], 0.095, "SBS Score Recorded")

        plt.ylim(0, 0.1)
        plt.xlabel('Minutes')
        plt.ylabel('Mean Amplitude Deviation')    
        plt.title(f'MAD at SBS={sbs_value} (5-min Double-Sided Window)')
    
        # save plots
        # folder_path = './AnalysisResults/Patient9_5MIN_DSW'
        # if not os.path.isdir(folder_path):
        #     os.makedirs(folder_path)
        # plt.savefig(os.path.join(folder_path, f'SBS_{sbs_value}_plot{count}.png'))
        plt.show()
        
        count += 1
        
def signal_vs_time(signal, sbs, window_size):
    if figure is None:
        figure = plt.figure(figsize=(8, 6))
    
    # Accelerometry Sampling Frequency
    freq = 100
    
    count = 0
    
    # Print Graph per SBS
    for i, sbs_value in enumerate(sbs):
        signal = signal[i,:]

        t = np.arange(0, len(signal), step=window_size) / (freq * 60)
        
        # Plot MAD against time
        plt.figure(figure.number)
        plt.plot(t, signal, color='blue')
        # SBS marker
        plt.axvline(t[len(t)//2], color='red', linestyle='--')
        plt.text(t[len(t)//2], 0.095, "SBS Score Recorded")

        plt.ylim(0, 150)
        plt.xlabel('Minutes')
        plt.ylabel('Heart Rate (bpm)')    
        plt.title(f'Heart Rate Signal vs Time at SBS={sbs_value}')

        plt.show()
        
        count += 1