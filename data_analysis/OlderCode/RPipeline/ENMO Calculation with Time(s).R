# library(adeptdata)
library(dplyr)
library(ggplot2)
library(reshape2)
library(lubridate)
library(readr)
library(read.gt3x)
library(scattermore)
library(zoo)

# Get GT3 Data into Data Frame
gt3xfile <-
  system.file(
    "extdata", "Patient6_Data.gt3x",
    package = "read.gt3x")

gt3xfolders <- unzip.gt3x("Patient6_Data.gt3x", location = tempdir())
gt3xfolder <- gt3xfolders[1]
X <- read.gt3x(gt3xfolder)

X <- as.data.frame(X)

# Code to shift TimeStamp
shift_duration <- days(1) + hours(21) + minutes(30)

Filter_Data <- X %>%
  mutate(newtime = time + shift_duration) %>%
  filter(time < min(newtime))
  
X[X$time >= "2023-11-29" & X$time <= "2023-11-30", ]

# Code to check TimeStamp Class
check_type <- function(data_frame, column_name, row_index) {
  value <- data_frame[row_index, column_name]
  type <- typeof(value)
  class_type <- class(value)
  
  cat("Value:", value, "\n")
  cat("Type:", type, "\n")
  cat("Class:", class_type, "\n")
}
check_type(X, "time", 1)


# Calculate ENMO 
X$Magnitude <- sqrt(X$X^2 + X$Y^2 + X$Z^2) - 1

#Make a Data Frame Object of Magnitudes
new_list <- data.frame(Magnitude = X$Magnitude)

#Calculate Moving Average Using Zoo Function
moving_average <- rollmean(new_list$Magnitude, k = 10000, fill = NA)

print(moving_average)

# Calculate Time in Seconds (X axis)
X$time_seconds <- seq(0, by = 1 / 100, length.out = nrow(X))

# Plot graph
ggplot(X, aes(x = time_seconds)) +
  geom_scattermore(aes(y = moving_average),
                   color = "purple", linetype = "solid") +
  labs(x = "Time [s]", y = "Magnitude [g]", color = "Magnitude",
       title = "100Hz Patient 6 ENMO Data") +
  scale_color_manual(values = c("purple"))

# Export plot to PNG
#ggsave("Patient1data1_magnitude_seconds.png", plot = accel_plot)