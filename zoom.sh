#!/bin/bash
# Example "Zoom" movie generation
# e.g. ./zoom.sh "A painting of zooming in to a surreal, alien world" Zoom.png 180

TEXT="$1"
FILENAME="$2"
MAX_EPOCHS=$3

LR=0.1
OPTIMISER=Adam
MAX_ITERATIONS=25
SEED=`shuf -i 1-9999999999 -n 1` # Keep the same seed each epoch for more deterministic runs

# Extract
FILENAME_NO_EXT=${FILENAME%.*}
FILE_EXTENSION=${FILENAME##*.}

# Initial run
python generate.py -p="$TEXT" -opt="$OPTIMISER" -lr=$LR -i=$MAX_ITERATIONS -se=$MAX_ITERATIONS --seed=$SEED -o="$FILENAME"
cp "$FILENAME" "$FILENAME_NO_EXT"-0000."$FILE_EXTENSION"
convert "$FILENAME" -distort SRT 1.01,0 -gravity center "$FILENAME"	# Zoom
convert "$FILENAME" -distort SRT 1 -gravity center "$FILENAME"	# Rotate

# Feedback image loop
for (( i=1; i<=$MAX_EPOCHS; i++ ))
do
  padded_count=$(printf "%04d" "$i")  
  python generate.py -p="$TEXT" -opt="$OPTIMISER" -lr=$LR -i=$MAX_ITERATIONS -se=$MAX_ITERATIONS --seed=$SEED -ii="$FILENAME" -o="$FILENAME"
  cp "$FILENAME" "$FILENAME_NO_EXT"-"$padded_count"."$FILE_EXTENSION"    
  convert "$FILENAME" -distort SRT 1.01,0 -gravity center "$FILENAME" # Zoom
  convert "$FILENAME" -distort SRT 1 -gravity center "$FILENAME"	# Rotate
done

# Make video - Nvidia GPU expected
ffmpeg -y -i "$FILENAME_NO_EXT"-%04d."$FILE_EXTENSION" -b:v 8M -c:v h264_nvenc -pix_fmt yuv420p -strict -2 -filter:v "minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=60'" video.mp4
