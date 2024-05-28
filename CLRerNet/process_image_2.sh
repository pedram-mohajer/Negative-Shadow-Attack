#!/bin/bash

# Function to process an image and save the result in section B

process_image() {
  image_file="$1"

  image_name=$(basename "$image_file")
  image_name="${image_name%.*}"
  image_folder="${image_name%.*}"

 
  mkdir -p out

  # Run the Python command with the image file and save the result in section B
  python3 data/run.py "$image_file" configs/clrernet/culane/clrernet_culane_dla34_ema.py clrernet_culane_dla34_ema.pth --out-file="out/$image_name.png"
  #mv "$image_file" "out/$image_folder/"
}


# Loop through all the images in the "demo" folder
for image in data/*.png; do
  process_image "$image"
done
