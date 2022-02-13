

import sys
import os



class TimelineData:
    prompt = []
    frames = []
    zoom = []


def read_timeline_file(path):

    timelineData = TimelineData()

    temp_prompt = "Blue Duck"
    temp_frames = 10
    temp_zoom = 1
    
    with open(path, 'r') as file:
        timeline_file_lines = [line.strip() for line in file]
    
    for line in timeline_file_lines:
        line = line.casefold().split(",")

        for part in line:
            part = part.strip()

            if part.startswith("prompt="):
                temp_prompt = part[7:].replace("\"", "").replace("\'", "")
                continue
            if part.startswith("frames="):
                temp_frames = int(part[7:])
                continue
            if part.startswith("zoom="):
                temp_zoom = float(part[5:])
                continue

        timelineData.prompt.append(temp_prompt)
        timelineData.frames.append(temp_frames)
        timelineData.zoom.append(temp_zoom)

    return timelineData


timeline = read_timeline_file("storyTest.txt")


print(timeline.prompt)
print(timeline.frames)
print(timeline.zoom)










