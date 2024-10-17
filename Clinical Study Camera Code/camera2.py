"""
This code captures video from camera 2 and saves it in chunks every 600 seconds. 
It also displays the captured video frames in real-time with a timestamp on each frame. 
The video files are saved in a directory named based on the current timestamp and a random number to ensure uniqueness
"""

# Import necessary libraries
import numpy as np
import cv2
import time
import os
import random
import sys
import datetime

# Define video parameters
fps = 24
width = 864
height = 640
video_codec = cv2.VideoWriter_fourcc("D", "I", "V", "X")

# Generate a random number to use in the filename
number = random.randint(0, 1000)

# Create a unique name for the directory based on the current timestamp and random number
name = str(time.time()) + '_patient_camera_2_'+ str(number)
print(name)

# Check if the directory with the same name already exists; if so, generate a new random number
if os.path.isdir(str(name)) is False:
    number = random.randint(0, 1000)
    # name = str(name)
    name = str(time.time()) + '_patient_camera_2_'+ str(number)

# Create the directory to store the video files
name = os.path.join(os.getcwd(), str(name))
print("ALl logs saved in dir:", name) 
os.mkdir(name)

# Initialize the camera capture
cap = cv2.VideoCapture(2)

# Set the camera properties to desired width and height
ret = cap.set(3, 864)
ret = cap.set(4, 480)

# Get the current directory path
cur_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Get the start time of the video capture
# start = time.time()
start = datetime.datetime.now().timestamp()

# Initialize variables for video file handling
video_file_count = 1
video_file = os.path.join(name, str(video_file_count) + "_camera_2_" + str(start) + ".avi") # can change here 
print("Capture video saved location : {}".format(video_file))

# Create a video write before entering the loop to save video chunks
video_writer = cv2.VideoWriter(
    video_file, video_codec, fps, (int(cap.get(3)), int(cap.get(4)))
)

# Enter the loop to start capturing video frames and displaying them
while cap.isOpened():
    # start_time = time.time()
    start_time = datetime.datetime.now().timestamp()
    ret, frame = cap.read()
    if ret == True:
        # Add a timestamp to the frame
        cv2.putText(frame,"Time: "+str(datetime.datetime.now().timestamp()),(10,30),2,0.8,(255,255,2))
        
        # Display the frame
        cv2.imshow("frame", frame)

        # Check if it's time to start a new video file (every 600 seconds)
        if datetime.datetime.now().timestamp() - start > 600:
            start = datetime.datetime.now().timestamp()
            video_file_count += 1

            # Generate a new filename for the next video chunk
            video_file = os.path.join(name, str(video_file_count)+ "_camera_2_" + str(start)  + ".avi") # can change here 
            video_writer = cv2.VideoWriter(
                video_file, video_codec, fps, (int(cap.get(3)), int(cap.get(4)))
            )

        # Write the frame to the current video writer
        video_writer.write(frame)

        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the camera and close all windows 
cap.release()
cv2.destroyAllWindows()