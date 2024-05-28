#!/bin/bash
INPUT_DIR="data/input"
OUTPUT_DIR="data/output"
PYTHON_SCRIPT="run_clrernet.py"
IMAGE_CONFIG="configs/clrernet/culane/clrernet_culane_dla34_ema.py"
WEIGHTS_PATH="clrernet_culane_dla34_ema.pth"
OUTPUT_FILE="result.png"

OVERLAY_FOLDERS="overlaid"
echo $(ls)
python "$PYTHON_SCRIPT" "$SCRIPT_DIR/INPUT_DIR" "$SCRIPT_DIR/$IMAGE_CONFIG" "$SCRIPT_DIR/$WEIGHTS_PATH" --out-file "$OUTPUT_FILE"