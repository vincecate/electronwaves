#!/usr/bin/python3
#
# requires
#   pip install moviepy

#!/usr/bin/python3
#
# requires
#   pip install moviepy

import os
import re
from PIL import Image
from moviepy.editor import ImageSequenceClip

# Directories
simulations_directory = "simulation"
cropped_directory = "cropped"

# Create the cropped directory if it doesn't exist
if not os.path.exists(cropped_directory):
    os.makedirs(cropped_directory)

# Function to extract the step number from the filename using regular expression
def extract_step_number(filename):
    match = re.search(r'wire_(\d+)\.png', filename)
    return int(match.group(1)) if match else 0

# Process and save the cropped images in the new directory
cropped_image_filenames = []
for filename in os.listdir(simulations_directory):
    if filename.endswith('.png') and filename.startswith('wire'):
        full_path = os.path.join(simulations_directory, filename)
        cropped_image_filenames.append(full_path)

# Sort the cropped files numerically by the step number
cropped_image_filenames.sort(key=lambda f: extract_step_number(os.path.basename(f)))

# Debug: Print sorted filenames
for filename in cropped_image_filenames:
    print(filename)

# Create a video clip from the image sequence
clip = ImageSequenceClip(cropped_image_filenames, fps=1)
clip.write_videofile('simulation_video.mp4')

