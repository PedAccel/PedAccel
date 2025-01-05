# library(adeptdata)
library(dplyr)
library(ggplot2)
library(reshape2)
library(lubridate)
library(readr)

# Replace "your_file.csv" with the path to your CSV file.
file_path <- "Patient1_Data1.csv"

# Read the CSV file and output it to the terminal
data <- read.csv(file_path, sep = ",")

is.data.frame(data)

print(data$Timestamp)



# ggplot(data, aes(Timestamp, Accelerometer.X)) + geom_line()

ggplot(data, aes(x = Timestamp)) +
  geom_line(aes(y = Accelerometer.X),
            color = "blue", linetype = "solid") +
  geom_line(aes(y = Accelerometer.Y),
            color = "red", linetype = "solid") +
  geom_line(aes(y = Accelerometer.Z),
            color = "green", linetype = "solid") +
  labs(x = "Time [s]", y = "Amplitude [g]", color = "Sensor\naxis",
       title = "100Hz Patient 1 Triaxial Data") +
  labs(color = "Sensor Axis") +  # Customize the legend title here
  scale_color_manual(values = c("blue", "red", "green"),
                     labels = c("X", "Y", "Z"))

# Export plot to PNG
#ggsave("Patient1data1.png", plot = accel_plot)