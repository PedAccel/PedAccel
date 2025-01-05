library(dplyr)
library(ggplot2)
library(reshape2)
library(lubridate)
library(readr)
library(read.gt3x)
library(scattermore)
library(zoo)

# TL;DR - Low Pass Filter for 12 hour time frame of Patient 6 Data
# ~ 11/29 8pm to 11/30 8am; adjust range based on interval of interest

# Get GT3 Data into Data Frame
gt3xfile <-
  system.file(
    "extdata", "Patient6_Data.gt3x",
    package = "read.gt3x")

gt3xfolders <- unzip.gt3x("Patient6_Data.gt3x", location = tempdir())
gt3xfolder <- gt3xfolders[1]
X <- read.gt3x(gt3xfolder)

X <- as.data.frame(X)

# Calculate Time in Seconds (X axis)
X$time_seconds <- seq(0, by = 1 / 100, length.out = nrow(X))

# Calculate ENMO
X$Magnitude <- sqrt(X$X^2 + X$Y^2 + X$Z^2) - 1

# Make a Data Frame Object of Magnitudes
new_list <- data.frame(Magnitude = X$Magnitude)

# ISOLATE RANGE
X <- X[208366:2592000,c(1:6)]

# Calculate Moving Average Using Zoo Function; Window Size = 10 seconds
moving_average <- rollmean(new_list$Magnitude, k = 1000, fill = NA)

print(moving_average)

# Plot graph with Moving Averages
options(repr.plot.width = 6, repr.plot.height =3)
ggplot(X, aes(x = time_seconds)) +
  geom_scattermore(aes(y = moving_average), # Change depending on Y
                   color = "purple", linetype = "solid") +
  labs(x = "Time [s]", y = "Magnitude [g]", color = "Magnitude",
       title = "100Hz Patient 6 ENMO Data") +
  scale_color_manual(values = c("purple")) +
  theme(
    plot.margin = margin(3, 1, 3, 1, "cm")  # Adjust dimensions of graph
  )

# Export plot to PNG
#ggsave("Patient1data1_magnitude_seconds.png", plot = accel_plot)