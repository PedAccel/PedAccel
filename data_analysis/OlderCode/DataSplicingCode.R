library(dplyr)
library(ggplot2)
library(reshape2)
library(lubridate)
library(readr)
library(read.gt3x)
library(scattermore)
library(tidyverse)
library(zoo)

# Read GTX File
gt3xfile <-
  system.file(
    "extdata", "Patient11_Data.gt3x",
    package = "read.gt3x")
gt3xfolders <- unzip.gt3x("Patient11_Data.gt3x", location = tempdir())
gt3xfolder <- gt3xfolders[1]
X <- read.gt3x(gt3xfolder)
X <- as.data.frame(X)

# Splice data between two date/times - *times are in GMT*
start_datetime <- as.POSIXct("2024-01-30 15:00:00")
end_datetime <- as.POSIXct("2024-02-03 11:00:00")

# Subset data - *change file path for your system*
subset_df <- subset(X, time >= start_datetime & time <= end_datetime)
file_path <- "C:/Users/sidha/PycharmProjects/PedAccel/Patient11_FullSBS.csv"

# Save the dataframe to a CSV file
write.csv(subset_df, file = file_path, row.names = FALSE)
