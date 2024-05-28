
#!/bin/bash


INPUT_DIR="data/input"
OUTPUT_DIR="data/output"

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
		python hybridnets_test.py -w weights/hybridnets.pth --source $attack_folder --output $overlay_output --show_seg True --imwrite False

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
            
	    python hybridnets_test.py -w weights/hybridnets.pth --source $folder --output $output --show_seg True --imwrite False

            ;;
        esac

        echo "Done"
        echo "----------------------------------------------"
    fi
done

