library(dplyr)
library(ggplot2)
library(reshape2)
library(lubridate)
library(readr)

# Replace "your_file.csv" with the path to your CSV file.
file_path <- "Patient1_Data1.csv"

# Read the CSV file and output it to the terminal
data <- read.csv(file_path, sep = ",")

# Calculate the vector magnitude and subtract 1
data$Magnitude <- sqrt(data$Accelerometer.X^2 + data$Accelerometer.Y^2 + data$Accelerometer.Z^2) -1

# Check the structure of your data frame to make sure the new variable is added
str(data)

# ggplot with the new Magnitude variable
ggplot(data, aes(x = Timestamp)) +
  geom_line(aes(y = Magnitude),
            color = "purple", linetype = "solid") +
  labs(x = "Time [s]", y = "Magnitude [g]", color = "Magnitude",
       title = "100Hz Patient 1 ENMO Data") +
  scale_color_manual(values = c("purple"))

# Export plot to PNG
#ggsave("Patient1data1_magnitude.png", plot = accel_plot)
