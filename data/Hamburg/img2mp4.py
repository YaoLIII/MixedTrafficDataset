# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:43:15 2024

@author: li
"""

import cv2
import glob

# Define image folder and output video file
image_folder = 'extracted_figs'
output_video = 'Hamburg_subset.mp4'

# Get list of images
images = sorted(glob.glob(f"{image_folder}/*.png"))

# Read the first image to get dimensions
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, 24, (width, height))

# Add each image to the video
for image in images:
    frame = cv2.imread(image)
    video.write(frame)

# Release the video writer
video.release()
cv2.destroyAllWindows()

'''or'''

from moviepy.editor import ImageSequenceClip

# Define the folder with images and output video file
image_folder = 'extracted_figs'
output_video = 'Hamburg_subset.mp4'

# Get a list of image file paths
images = sorted(glob.glob(f"{image_folder}/*.png"))

# Create a clip from the image sequence
clip = ImageSequenceClip(images, fps=24)

# Write the clip to a file
clip.write_videofile(output_video, codec='libx264')
