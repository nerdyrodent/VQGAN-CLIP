import subprocess
import os
from PIL import Image
from pathlib import Path

print ("1st call#######")
# subprocess.call("python generate.py -p 'A painting of an apple in a fruit bowl' -o ", shell=True)

print ("2nd call#######")
# subprocess.call("python generate.py -p 'A painting of an apple in a fruit bowl'", shell=True)



root_dir = "./bing/dataset_output"
for subdir, dirs, files in os.walk(root_dir):
  for file in files:
    path = os.path.join(subdir, file)
    outdir = os.path.join(subdir, 'output1/')
    outpath = os.path.join(subdir, 'output1/',file)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if os.path.exists(outpath):
      print('-')
    else:
      # print(path)
      # print(outpath)
      city = subdir.split("/")[-1]
      try:
        path = os.path.join(subdir, file)
        command_str = "python generate.py -p 'an illustration of colorful " + city +" on a sunny day' -o " + outpath + " -ii " + path + " "
        # print(command_str)
        subprocess.call(command_str, shell=True)
      except Exception as e:
        print(e)
        print("exception for:"+outpath)
