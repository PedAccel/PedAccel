# library(adeptdata)
library(dplyr)
library(ggplot2)
library(reshape2)
library(lubridate)
library(readr)

# Replace "your_file.csv" with the path to your CSV file.
file_path <- "Patient1_Data1.csv"

# Read the CSV file and output it to the terminal
myData <- read.csv(file_path, sep = ",")

#Block 1
# Assuming myData is a data frame in R
lengthData <- myData$`Accelerometer Y`
N <- length(lengthData)
cat("N:", N, "\n")
# Assuming 'f' is defined before this code
f <-  # Assign the value of 'f' here
  timeArray1 <- numeric(N)
for (i in 1:N) {
  timeArray1[i] <- i * (1 / f)
}
cat("Length of timeArray1:", length(timeArray1), "\n")
myData$`Time(s)1` <- timeArray1
cat("Length of timeArray1 after adding to myData:", length(timeArray1), "\n")
t <- timeArray1[N]
cat("t:", t, "\n")
print(myData)
#Block 2
# Assuming 'f' and 't' are defined before this code
Frequency <- paste(f, "Hz", sep = "")
min <- t / 60
# No justification for these specific windows, they just make the graph look good
# Rolling Peak to Peak
WindowLengthPeaks <- 0.1 * min
# Average has to be shorter than peak to peak so zeros don't weigh it down
# Rolling Average
WindowLengthAvg <- 0.1 * min
# This gives 150 total data points
# Peak to Peak
interval <- 0.4 * min
#Block 3
# Parameters
time <- t
frequency <- f
interval <- interval
Slices <- 150
# Variables
peakArray <- numeric()
avgArray <- numeric()
timeArray <- numeric()
t1 <- 0
t2 <- t1 + interval
count <- 0
# Code
# Splice the Data into 5-second intervals
for (i in seq(0, Slices - 1, by = 1)) {
  SlicedData <- myData[myData$`Time(s)1` >= t1 & myData$`Time(s)1` <= t2, ]
  
  # Compute vector magnitude of the spliced interval
  y <- (SlicedData$`Accelerometer X`^2 + SlicedData$`Accelerometer Y`^2 + SlicedData$`Accelerometer Z`^2)
  ydata <- sqrt(y) - 1
  
  # Find the minimum vector magnitude
  min_val <- min(ydata)
  # Find the maximum vector magnitude
  max_val <- max(ydata)
  PeaktoPeak <- max_val - min_val
  
  # Create array of peaks for the graph
  peakArray <- c(peakArray, PeaktoPeak)
  
  avg <- mean(ydata)
  avgArray <- c(avgArray, avg)
  
  # Create array of times for the graph
  timeArray <- c(timeArray, t1)
  
  t1 <- t1 + interval
  t2 <- t2 + interval
}
#Block 4
# Assuming you have already defined 'ydata', 'xdata', 'Frequency', and 'part'
# Plot Data
plot(xdata, ydata, xlab = "Time(seconds)", ylab = "Peak to Peak Amplitude(g)",
     main = paste(Frequency, part, "Peak to Peak Magnitudes", sep = " "),
     pch = 16, col = "blue")
# Add error bars
# (Note: R's base plot function does not have a direct equivalent to Python's 'errorbar' function,
# so I'm assuming you want to plot points with error bars represented by vertical lines.)
arrows(x0 = xdata, y0 = ydata - 0.5, x1 = xdata, y1 = ydata + 0.5, angle = 90, code = 3, length = 0.1)
# Add legend if needed
# legend("topright", legend = "Original Points", pch = 16, col = "blue")