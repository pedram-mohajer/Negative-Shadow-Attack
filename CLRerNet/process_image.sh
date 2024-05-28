#!/bin/bash


INPUT_DIR="data/input"
OUTPUT_DIR="data/output"


PYTHON_SCRIPT="demo/image_demo.py"
IMAGE_CONFIG="configs/clrernet/culane/clrernet_culane_dla34_ema.py"
WEIGHTS_PATH="clrernet_culane_dla34_ema.pth"
OUTPUT_FILE="result.png"

OVERLAY_FOLDERS="overlaid"

for folder in "$INPUT_DIR"/*; do
    if [ -d "$folder" ]; then
        # Get the folder name
        folder_name=$(basename -- "$folder")

        case $folder_name in 
            *$OVERLAY_FOLDERS* )
            echo "Found overlay folder"
            for attack_folder in "$folder"/*; do
                attack_name=$(basename -- "$attack_folder")
                overlay_output="$OUTPUT_DIR/$folder_name/$attack_name"

                echo "overlay output: $overlay_output"
                if [ -d "$overlay_output" ]; then
                    # If it exists, delete it
                    echo "Delete $overlay_output" 
                    rm -rf "$overlay_output"
                fi
                mkdir -p "$overlay_output"

                for image in "$attack_folder"/*; do
                    if [ -f "$image" ]; then

                        image_name=$(basename -- "$image")
                        echo "$overlay_output/$image_name"
                        python "$PYTHON_SCRIPT" "$image" "$IMAGE_CONFIG" "$WEIGHTS_PATH" --out-file "$overlay_output/$image_name"
                        #echo "Processed $output/$image_name"
                    fi
                    
                done
            done
            ;;
            *)

            output="$OUTPUT_DIR/$folder_name"
            echo "output : $output"

            if [ -d "$output" ]; then
                # If it exists, delete it
                echo "Delete $output" 
                rm -rf "$output"
            fi

            mkdir -p "$output"

            for image in "$folder"/*; do

                if [ -f "$image" ]; then

                    image_name=$(basename -- "$image")

                    python "$PYTHON_SCRIPT" "$image" "$IMAGE_CONFIG" "$WEIGHTS_PATH" --out-file "$output/$image_name"
                    #echo "Processed $output/$image_name"
                fi
            done

            ;;
        esac

        echo "Done"
        echo "----------------------------------------------"
    fi
done
