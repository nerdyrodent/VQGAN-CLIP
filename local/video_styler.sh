#!/bin/bash
# Video styler - Use all images in a directory and style them
# video_styler.sh video.mp4

# Style text
TEXT="Oil painting of a woman in the foreground | pencil art landscape background"

## Input and output frame directories
FRAMES_IN="/home/nerdy/github/VQGAN-CLIP/VideoFrames"
FRAMES_OUT="/home/nerdy/github/VQGAN-CLIP/Saves/VideoStyleTesting"

## Output image size
HEIGHT=640
WIDTH=360

## Iterations
ITERATIONS=25
SAVE_EVERY=$ITERATIONS

## Optimiser & Learning rate
OPTIMISER=Adagrad	# Adam, AdamW, Adagrad, Adamax
LR=0.2

# Fixed seed
SEED=`shuf -i 1-9999999999 -n 1` # Keep the same seed each frame for more deterministic runs

# MAIN
############################
mkdir -p "$FRAMES_IN"
mkdir -p "$FRAMES_OUT"

# For cuDNN determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Extract video into frames
ffmpeg -y -i "$1" -q:v 2 "$FRAMES_IN"/frame-%04d.jpg

# Style all the frames
ls "$FRAMES_IN" | while read file; do
   # Set the output filename
   FILENAME="$FRAMES_OUT"/"$file"-"out".jpg
   
   # And imagine!
   echo "Input frame: $file"
   echo "Style text: $TEXT"
   echo "Output file: $FILENAME"

   python generate.py -p "$TEXT" -ii "$FRAMES_IN"/"$file" -o "$FILENAME" -opt "$OPTIMISER" -lr "$LR" -i "$ITERATIONS" -se "$SAVE_EVERY" -s "$HEIGHT" "$WIDTH" -sd "$SEED" -d True
done

ffmpeg -y -i "$FRAMES_OUT"/frame-%04d.jpg-out.jpg -b:v 8M -c:v h264_nvenc -pix_fmt yuv420p -strict -2 -filter:v "minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=60'" style_video.mp4
