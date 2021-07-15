#!/bin/bash

# Using each optimiser, generate images using a range of learning rates
# Produce a labelled montage to easily view the results

TEXT="A painting in the style of Paul Gauguin"
OUT_DIR="/home/nerdy/github/VQGAN-CLIP/Saves/OptimiserTesting-60it-Noise-NPW-1"
ITERATIONS=60
SAVE_EVERY=60
HEIGHT=256
WIDTH=256
SEED=`shuf -i 1-9999999999 -n 1` # Keep the same seed each epoch for more deterministic runs

# Main
#################

export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$OUT_DIR"

function do_optimiser_test () {
  OPTIMISER="$1"
  LR="$2"
  STEP="$3"
  NPW="$4"
  for i in {1..10}
  do
    PADDED_COUNT=$(printf "%03d" "$COUNT")
    echo "Loop for $OPTIMISER - $LR"
    python generate.py -p "$TEXT" -in pixels -o "$OUT_DIR"/"$PADDED_COUNT"-"$OPTIMISER"-"$LR"-"$NPW".png -opt "$OPTIMISER" -lr "$LR" -i "$ITERATIONS" -se "$SAVE_EVERY" -s "$HEIGHT" "$WIDTH" --seed "$SEED" -d True -iw 1 -nps 666 -npw "$NPW" -d True
    LR=$(echo $LR + $STEP | bc)
    ((COUNT++))
  done
}

# Test optimisers
COUNT=0
do_optimiser_test "Adam" .1 .1 1
COUNT=10
do_optimiser_test "AdamW" .1 .1 1
COUNT=20
do_optimiser_test "Adamax" .1 .1 1
COUNT=30
do_optimiser_test "Adagrad" .1 .25 1
COUNT=40
do_optimiser_test "AdamP" .1 .25 1
COUNT=50
do_optimiser_test "RAdam" .1 .25 1
COUNT=60
do_optimiser_test "DiffGrad" .1 .25 1

# Make montage
mogrify -font Liberation-Sans -fill white -undercolor '#00000080' -pointsize 14 -gravity NorthEast -annotate +10+10 %t "$OUT_DIR"/*.png
montage "$OUT_DIR"/*.png -geometry 256x256+1+1 -tile 10x7 collage.jpg
