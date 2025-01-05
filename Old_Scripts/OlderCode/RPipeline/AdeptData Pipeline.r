library(adeptdata)
library(dplyr)
library(ggplot2)
library(reshape2)
library(lubridate)
library(readr)

# Replace "your_file.csv" with the path to your CSV file.
file_path <- "Patient1_Data.csv"

# Read the CSV file and output it to the terminal
data <- read.csv(file_path) #, headers = TRUE, stringsAsFactors = FALSE)

# Delete rows with junk information
data <- data[-(1:9), ]

# Check the column names to ensure they match the expected names
# colnames(new_data)

# Check the structure of 'new_data'
View(data)

# Check the first few rows of 'new_data' to inspect the data
# head(new_data, 6)


# Convert the 'Timestamp' column to POSIXct format
# Keep in same data frame
# new_data$Timestamps <- as.POSIXct(new_data$Timestamps, format = "%m/%d/%Y %I:%M:%OS %p")

# print(new_data)



# print(new_data)

# column_number <- 3
# Timestamp <- new_data[, 0]
# Accelerometer_X <- new_data[, 1]
# Accelerometer_Y <- new_data[, 2]
# Accelerometer_Z <- new_data[, 3]

# matplot(new_data[,1], new_data[, -1], type = "l")

# print(new_data)

# exists(new_data$Timestamp)

# class(new_data)

# plot(x = new_data$Timestamp, y = new_data$"Accelerometer X", 
#     xlab = "Time", ylab = "X Accel", main = "Graph1")

# accel <- read.csv("2023-24/RawAccel_Sample1.csv") %>%
# select(Timestamp, "Accelerometer X", "Accelerometer Y", "Accelerometer Z")

# accel_plot <- ggplot(data = accel, aes(x = Timestamp, color = variable)) +
# geom_line(data = accel, aes(y = Accelerometer_X),
# color = "blue", linetype = "solid") +
# geom_line(data = accel, aes(y = Accelerometer_Y),
# color = "red", linetype = "solid") +
# geom_line(data = accel, aes(y = Accelerometer_Z),
# color = "green", linetype = "solid") +
# labs(x = "Time [s]", y = "Amplitude [g]", color = "Sensor\naxis",
# title = "100Hz Patient 1 Triaxial Data") +
# labs(color = "Sensor Axis") +  # Customize the legend title here
# scale_color_manual(values = c("blue", "red", "green"),
# labels = c("X", "Y", "Z"))

# #Export plot to PNG
# ggsave("Patient1data1.png", plot = accel_plot)