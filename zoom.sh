#!/bin/bash

TEXT="$1"
FILENAME="$2"

LR=0.1
MAX_ITERATIONS=25
MAX_EPOCHS=600
SEED=1092731 # Keep the same seed each epoch for more deterministic runs

# Extract
FILENAME_NO_EXT=${FILENAME%.*}
FILE_EXTENSION=${FILENAME##*.}

# Initial run
python generate.py -p="$TEXT" -lr=$LR -i=$MAX_ITERATIONS -se=$MAX_ITERATIONS --seed=$SEED -o="$FILENAME"
cp "$FILENAME" "$FILENAME_NO_EXT-0000.$FILE_EXTENSION"
convert "$FILENAME" -distort SRT 1.01,0 -gravity center "$FILENAME"

# Now convert and feedback
for (( i=1; i<=$MAX_EPOCHS; i++ ))
do
  padded_count=$(printf "%04d" "$i")  
  python generate.py -p="$TEXT" -lr=$LR -i=$MAX_ITERATIONS -se=$MAX_ITERATIONS --seed=$SEED -ii="$FILENAME" -o="$FILENAME"
  cp "$FILENAME" "$FILENAME_NO_EXT-$padded_count.$FILE_EXTENSION"    
  convert "$FILENAME" -distort SRT 1.01,0 -gravity center "$FILENAME"
done
