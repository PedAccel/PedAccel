#This Script Generates histograms periods of movement vs non movement

#Import Necessary Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random

from Modules import Actigraph_Metrics, General_Functions

if __name__ == "__main__":

#Data has been carefully sliced for instances of no interference where the patient has varying degrees of movement

#Import sleep data,sliced in R
    sleep1 = pd.read_csv("Patient5MovementAnalysis/Probably_sleepingNM1.csv")
    sleep2 = pd.read_csv("Patient5MovementAnalysis/Probably_sleepingNM2.csv")
    sleep3 = pd.read_csv("Patient5MovementAnalysis/Probably_sleepingNM3.csv")

#Append sleep data to 1 data frame
    total_sleeping = pd.concat([sleep1,sleep2, sleep3])
    print('total sleeping df')
    print(total_sleeping.head(5))

#Import non-moving awake data, sliced in R
    awake1_NM = pd.read_csv("Patient5MovementAnalysis/Probably_awakeNM1.csv")
    awake2_NM = pd.read_csv("Patient5MovementAnalysis/Probably_awakeNM2.csv")

#Append non moving awake data to 1 data frame
    total_waking_NM = pd.concat([awake1_NM, awake2_NM])
    print('total waking non-movement df')
    print(total_waking_NM.head(5))

#Append total non-moving data
    total_non_moving = pd.concat([total_sleeping, total_waking_NM])

#Import moving awake data, sliced in R
    awake1 = pd.read_csv("Patient5MovementAnalysis/Awake_moving1.csv")
    awake2 = pd.read_csv("Patient5MovementAnalysis/Awake_moving2.csv")

#Append awake data to 1 data frame
    total_waking = pd.concat([awake1, awake2])
    print('total waking df')
    print(total_waking.head(5))


#Import transition states data
    Waking_Up = pd.read_csv("Patient5MovementAnalysis/Sleep_to_awake1.csv")
    In_Out_of_sleep = pd.read_csv("Patient5MovementAnalysis/InAndOutOfSleep1.csv")

# Note: Other transition periods ignored for now

#Create New Columns for sleeping data
    General_Functions.create_time_column(total_sleeping)
    General_Functions.create_absMag_Column(total_sleeping)
#Create New Columns for non-movement waking data
    General_Functions.create_time_column(total_waking_NM)
    General_Functions.create_absMag_Column(total_waking_NM)
#Create New Columns for waking data
    General_Functions.create_time_column(total_waking)
    General_Functions.create_absMag_Column(total_waking)
#Create New Columns for total non moving data
    General_Functions.create_time_column(total_non_moving)
    General_Functions.create_absMag_Column(total_non_moving)

#Create New Columns for transitioning data
    General_Functions.create_time_column(Waking_Up)
    General_Functions.create_absMag_Column(Waking_Up)

    General_Functions.create_time_column(In_Out_of_sleep)
    General_Functions.create_absMag_Column(In_Out_of_sleep)

#Plot Raw Data for total sleeping
    General_Functions.plot_Data(total_sleeping,'Time(s)', 'VecMag',title = 'total_sleeping')

#Plot Raw Data for total waking
    General_Functions.plot_Data(total_waking, 'Time(s)', 'VecMag', title = 'total_waking')

#Plot Raw Data for total non moving
    General_Functions.plot_Data(total_non_moving, 'Time(s)', 'VecMag', title='total_non_moving')


#Normalize data so that the same number of points are plotted
    ysleeping = total_sleeping['VecMag']
    ywaking_NM = total_waking_NM['VecMag']
    ywaking = total_waking['VecMag']
    y_NM = total_non_moving['VecMag']
    myList = [ysleeping, ywaking_NM, ywaking, y_NM]
    min = len(myList[0])
    for index in range(len(myList)):
        if len(myList[index]) < min:
            min = len(myList[index])

    ysleeping = random.sample(list(total_sleeping['VecMag']), min)
    ywaking_NM = random.sample(list(total_waking_NM['VecMag']), min)
    ywaking = random.sample(list(total_waking['VecMag']), min)
    y_NM = random.sample(list(total_non_moving['VecMag']), min)

#Plot Histograms of Raw Data in side to side format, moving vs non-moving
#show two histograms in one figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    x1 = ywaking
    x2 = y_NM
# Assign colors, names, bins to 1st histogram
    bins1 = [.8,.9,.95,1, 1.04,1.08,1.12,1.2, 1.4]
    colors = ['red', 'blue']
    labels = ['Awake and Moving', 'Either Awake or Sleeping, but not moving']
# Make the histogram using a list of lists
    ax[0].hist([x1,x2], bins=bins1, color=colors, label=labels)
# Assign colors, names, bins to 2nd histogram
    bins2 = [1.4,1.5,1.6,1.7,1.8,1.9,2.2,2.4,2.8,3.2,4,5,6,7,8,9]
#make 2nd histogram
    ax[1].hist([x1,x2], bins=bins2, color=colors, label=labels)

# Plot formatting
#title the plots
    ax[0].set_title('Histogram of movement vs non-movement\nperiods with raw data')
    ax[1].set_title('Histogram of movement vs non-movement\nperiods with raw data and larger accelerations')
    ax[0].set_xlabel('Accelerations between .9g and 1.4g')
    ax[0].set_ylabel('Frequency')
    ax[1].set_xlabel('Accelerations above 1.4g')
    ax[1].set_ylabel('Frequency')
    ax[0].legend()
    ax[1].legend()
    plt.show()

#Plot Histograms of Raw Data in side to side format, sleep vs awake
#show two histograms in one figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    x1 = ywaking
    x2 = ywaking_NM
    x3 = ysleeping
# Assign colors, names, bins to 1st histogram
    bins1 = [.8,.9,.95,1, 1.04,1.08,1.12,1.2, 1.4]
    colors = ['red', 'orange','blue']
    labels = ['Awake and Moving', 'Probably Awake, but not moving', 'Probably Sleeping']
# Make the histogram using a list of lists
    ax[0].hist([x1,x2,x3], bins=bins1, color=colors, label=labels)
# Assign colors, names, bins to 2nd histogram
    bins2 = [1.4,1.5,1.6,1.7,1.8,1.9,2.2,2.4,2.8,3.2,4,5,6,7,8,9]
#make 2nd histogram
    ax[1].hist([x1,x2,x3], bins=bins2, color=colors, label=labels)

# Plot formatting
#title the plots
    ax[0].set_title('Histogram of sleeping and wakin\n periods with raw data')
    ax[1].set_title('Histogram of sleeping and waking\nperiods with raw data and larger accelerations')
    ax[0].set_xlabel('Accelerations between .9g and 1.4g')
    ax[0].set_ylabel('Frequency')
    ax[1].set_xlabel('Accelerations above 1.4g')
    ax[1].set_ylabel('Frequency')
    ax[0].legend()
    ax[1].legend()
    plt.show()

    y_in_out = In_Out_of_sleep['VecMag']
    myList.append(y_in_out)
    min = len(myList[0])
    for index in range(len(myList)):
        if len(myList[index]) < min:
            min = len(myList[index])

    ysleeping = random.sample(list(total_sleeping['VecMag']), min)
    ywaking_NM = random.sample(list(total_waking_NM['VecMag']), min)
    ywaking = random.sample(list(total_waking['VecMag']), min)
    y_NM = random.sample(list(total_non_moving['VecMag']), min)
    y_in_out = random.sample(list(In_Out_of_sleep['VecMag']), min)

#Plot Histograms of Raw Data in side to side format, sleep vs awake vs in and out
#show two histograms in one figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    x1 = ywaking
    x2 = ywaking_NM
    x3 = ysleeping
    x4 = y_in_out
# Assign colors, names, bins to 1st histogram
    bins1 = [.8,.9,.95,1, 1.04,1.08,1.12,1.2, 1.4]
    colors = ['red', 'orange','blue', 'green']
    labels = ['Awake and Moving', 'Probably Awake, but not moving', 'Probably Sleeping','In and out of sleep']
# Make the histogram using a list of lists
    ax[0].hist([x1,x2,x3,x4], bins=bins1, color=colors, label=labels)
# Assign colors, names, bins to 2nd histogram
    bins2 = [1.4,1.5,1.6,1.7,1.8,1.9,2.2,2.4,2.8,3.2,4,5,6,7,8,9]
#make 2nd histogram
    ax[1].hist([x1,x2,x3,x4], bins=bins2, color=colors, label=labels)

# Plot formatting
#title the plots
    ax[0].set_title('Histogram of sleeping and waking periods\nwith raw data, in and out of sleep included')
    ax[1].set_title('Histogram of sleeping and waking periods\nwith raw data, in and out of sleep included,\nand larger accelerations')
    ax[0].set_xlabel('Accelerations between .9g and 1.4g')
    ax[0].set_ylabel('Frequency')
    ax[1].set_xlabel('Accelerations above 1.4g')
    ax[1].set_ylabel('Frequency')
    ax[0].legend()
    ax[1].legend()
    plt.show()


    """
MAD Data Histograms
    """
#MAD sleeping Data
    myX, MAD_sleeping = Actigraph_Metrics.skdh_MAD(total_sleeping, wlen=50)

#MAD waking data
    myX, MAD_waking = Actigraph_Metrics.skdh_MAD(total_waking, wlen=50)

    myX, MAD_waking_NM = Actigraph_Metrics.skdh_MAD(total_waking_NM, wlen=50)

    myX, MAD_non_moving = Actigraph_Metrics.skdh_MAD(total_non_moving, wlen=50)

    myX, MAD_In_Out = Actigraph_Metrics.skdh_MAD(In_Out_of_sleep, wlen=50)

    myList = [MAD_sleeping, MAD_waking, MAD_waking_NM, MAD_non_moving]
    min = len(myList[0])
    for index in range(len(myList)):
        if len(myList[index]) < min:
            min = len(myList[index])

    MAD_waking = random.sample(list(MAD_waking), min)
    MAD_waking_NM = random.sample(list(MAD_waking_NM), min)
    MAD_sleeping = random.sample(list(MAD_sleeping), min)
    MAD_non_moving = random.sample(list(MAD_non_moving), min)



#Plot Histograms of Raw Data in side to side format, moving vs non-moving MAD
#show two histograms in one figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    x1 = MAD_waking
    x2 = MAD_non_moving
# Assign colors, names, bins to 1st histogram
    bins1 = [0, .02,.04, .06, .1, .15, .2]
    bins2 = [.2,.3,.4, .5, .8, 1, 1.3, 1.5]
    colors = ['red', 'blue']
    labels = ['Awake and Moving', 'Either Awake or Sleeping, but not moving']
# Make the histogram using a list of lists
    ax[0].hist([x1,x2], bins=bins1, color=colors, label=labels)
# Assign colors, names, bins to 2nd histogram
#make 2nd histogram
    ax[1].hist([x1,x2], bins=bins2, color=colors, label=labels)

# Plot formatting
#title the plots
    ax[0].set_title('Histogram of movement vs non-movement periods\nwith MAD data')
    ax[1].set_title('Histogram of movement vs non-movement periods\nwith MAD data and larger accelerations')
    ax[0].set_xlabel('MAD between 0 and .2')
    ax[0].set_ylabel('Frequency')
    ax[1].set_xlabel('MAD above .2')
    ax[1].set_ylabel('Frequency')
    ax[0].legend()
    ax[1].legend()
    plt.show()

#Plot Histograms of Raw Data in side to side format, sleep vs awake
#show two histograms in one figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    x1 = MAD_waking
    x2 = MAD_waking_NM
    x3 = MAD_sleeping
# Assign colors, names, bins to 1st histogram
    bins1 = [0, .02,.04, .06, .1, .15, .2]
    bins2 = [.2,.3,.4, .5, .8, 1, 1.3, 1.5]
    colors = ['red', 'orange','blue']
    labels = ['Awake and Moving', 'Probably Awake, but not moving', 'Probably Sleeping']
# Make the histogram using a list of lists
    ax[0].hist([x1,x2,x3], bins=bins1, color=colors, label=labels)
# Assign colors, names, bins to 2nd histogram
#make 2nd histogram
    ax[1].hist([x1,x2,x3], bins=bins2, color=colors, label=labels)

# Plot formatting
#title the plots
    ax[0].set_title('Histogram of movement vs non-movement periods\nwith MAD data')
    ax[1].set_title('Histogram of movement vs non-movement periods\nwith MAD data and larger accelerations')
    ax[0].set_xlabel('MAD between 0 and .2')
    ax[0].set_ylabel('Frequency')
    ax[1].set_xlabel('MAD above .2')
    ax[1].set_ylabel('Frequency')
    ax[0].legend()
    ax[1].legend()
    plt.show()

    """
    Update MAD with In and Out of sleep
    """
    y_in_out = MAD_In_Out
    myList.append(y_in_out)
    min = len(myList[0])
    for index in range(len(myList)):
        if len(myList[index]) < min:
            min = len(myList[index])

    MAD_waking = random.sample(list(MAD_waking), min)
    MAD_waking_NM = random.sample(list(MAD_waking_NM), min)
    MAD_sleeping = random.sample(list(MAD_sleeping), min)
    MAD_non_moving = random.sample(list(MAD_non_moving), min)
    MAD_In_Out = random.sample(list(MAD_In_Out), min)

#Plot Histograms of Raw Data in side to side format, sleep vs awake vs in and out, MAD
#show two histograms in one figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    x1 = MAD_waking
    x2 = MAD_waking_NM
    x3 = MAD_sleeping
    x4 = MAD_In_Out
# Assign colors, names, bins to 1st histogram
    bins1 = [0, .02, .04, .06, .1, .15, .2]
    bins2 = [.2, .3, .4, .5, .8, 1, 1.3, 1.5]
    colors = ['red', 'orange','blue', 'green']
    labels = ['Awake and Moving', 'Probably Awake, but not moving', 'Probably Sleeping','In and out of sleep']
# Make the histogram using a list of lists
    ax[0].hist([x1,x2,x3,x4], bins=bins1, color=colors, label=labels)
# Assign colors, names, bins to 2nd histogram
#make 2nd histogram
    ax[1].hist([x1,x2,x3,x4], bins=bins2, color=colors, label=labels)

# Plot formatting
#title the plots
    ax[0].set_title('Histogram of movement vs non-movement periods\nwith MAD data')
    ax[1].set_title('Histogram of movement vs non-movement periods\nwith MAD data and larger accelerations')
    ax[0].set_xlabel('MAD between 0 and .2')
    ax[0].set_ylabel('Frequency')
    ax[1].set_xlabel('MAD above .2')
    ax[1].set_ylabel('Frequency')
    ax[0].legend()
    ax[1].legend()
    plt.show()

#Plot transition states of Raw Data
    plt.plot(Waking_Up['Time(s)'], Waking_Up['VecMag'])
    plt.title('Waking Up Transition raw Data,\npatient barely wakes up')
    plt.axvline(x = 340, color = 'r', label = 'axvline - full height')
    plt.axvline(x=380, color='r', label='axvline - full height')
    plt.show()


#Generate MAD data for Transition states
    myXWU, MAD_WU = Actigraph_Metrics.MAD(Waking_Up, wlen=50)

#Plot transition states of MAD Data
    plt.plot(myXWU, MAD_WU)
    plt.title('Waking Up Transition MAD data,\npatient barely wakes up')
    # only one line may be specified; full height
    plt.axvline(x = 680, color = 'r', label = 'axvline - full height')
    plt.axvline(x=760, color='r', label='axvline - full height')
    plt.show()