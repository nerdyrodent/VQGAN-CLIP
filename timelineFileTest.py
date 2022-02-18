# Timeline tool made by Henry Turbedsky (https://github.com/HenryTurbedsky), for use with VQGAN-CLIP (https://github.com/nerdyrodent/VQGAN-CLIP)
# This program is for testing a timeline file to see if its being read in properly


import sys
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-tf",    "--timeline_file", type=str, help="File with timeline", default="samples/timeline/timelineTest.txt", dest='timeline_file')
args = parser.parse_args()

# All the data from the timeline file is read into this class
class TimelineData:
    prompt = []
    frames = []
    zoom = []
    iterations = []
    zoom_shift_x = []
    zoom_shift_y = []

# Takes in a path to a timeline file and outputs a instance of the (TimelineData) class
def read_timeline_file(path):

    timelineData = TimelineData()

    # set the default values
    temp_prompt = "Blue Duck"
    temp_frames = 10
    temp_zoom = 1
    temp_iterations = 10
    temp_shift_x = 0
    temp_shift_y = 0

    # Opens the timeline file
    with open(path, 'r') as file:
        timeline_file_lines = [line.strip() for line in file]
    
    for line in timeline_file_lines:
        # Skip if a line starts with the symbol "#", can be used to comment things out.
        if line.startswith("#"):
            continue

        line = line.casefold().split(",")

        for part in line:
            part = part.strip()
            
            if part.startswith("prompt="):
                temp_prompt = part[7:].replace("\"", "").replace("\'", "")
                continue
            if part.startswith("frames="):
                temp_frames = int(part[7:])
                continue
            if part.startswith("iterations="):
                temp_iterations = int(part[11:])
                continue
            if part.startswith("zoom="):
                temp_zoom = float(part[5:])
                continue
            if part.startswith("xshift="):
                temp_shift_x = int(part[7:])
                continue
            if part.startswith("yshift="):
                temp_shift_y = int(part[7:])
                continue

        
        # Skips the line since the settings make the line useless
        if temp_frames <= 0 or temp_iterations <= 0:
            print("Warning: Timeline: (frames=, iterations=) cant be set as non positive integers: ", line)
            continue

        timelineData.prompt.append(temp_prompt)
        timelineData.frames.append(temp_frames)
        timelineData.zoom.append(temp_zoom)
        timelineData.iterations.append(temp_iterations)
        timelineData.zoom_shift_x.append(temp_shift_x)
        timelineData.zoom_shift_y.append(temp_shift_y)

    return timelineData


timeline = read_timeline_file(args.timeline_file)

print(f"\n===== Timeline reading in file: {args.timeline_file} =====\n")
print("Prompts: ", timeline.prompt)
print("Frames: ", timeline.frames)
print("Zoom: ", timeline.zoom)
print("X Shift: ", timeline.zoom_shift_x)
print("Y Shift: ", timeline.zoom_shift_y)









