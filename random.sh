#!/bin/bash

text_one=("A painting of a" "A pencil art sketch of a" "An illustration of a" "A photograph of a")
text_two=("spinning" "dreaming" "watering" "loving" "eating" "drinking" "sleeping" "repeating" "surreal" "psychedelic")
text_three=("fish" "egg" "peacock" "watermelon" "pickle" "horse" "dog" "house" "kitchen" "bedroom" "door" "table" "lamp" "dresser" "watch" "logo" "icon" "tree"
 "grass" "flower" "plant" "shrub" "bloom" "screwdriver" "spanner" "figurine" "statue" "graveyard" "hotel" "bus" "train" "car" "lamp" "computer" "monitor")
styles=("Art Nouveau" "Camille Pissarro" "Michelangelo Caravaggio" "Claude Monet" "Edgar Degas" "Edvard Munch" "Fauvism" "Futurism" "Impressionism"
 "Picasso" "Pop Art" "Modern art" "Surreal Art" "Sandro Botticelli" "oil paints" "watercolours" "weird bananas" "strange colours")

pickword() {
   local array=("$@")
   ARRAY_RANGE=$((${#array[@]}-1))
   RANDOM_ENTRY=`shuf -i 0-$ARRAY_RANGE -n 1`   
   UPDATE=${array[$RANDOM_ENTRY]}
}


# Generate some images
for number in {1..50}
do
   # Make some random text
   pickword "${text_one[@]}"
   TEXT=$UPDATE
   pickword "${text_two[@]}"
   TEXT+=" "$UPDATE
   pickword "${text_three[@]}"
   TEXT+=" "$UPDATE
   pickword "${text_three[@]}"
   TEXT+=" and a "$UPDATE   
   pickword "${styles[@]}"
   TEXT+=" in the style of "$UPDATE
   pickword "${styles[@]}"
   TEXT+=" and "$UPDATE

   python generate.py -p "$TEXT" -o "$number".png
done


