# Project Configuration Guide

This guide outlines the necessary steps for setting up the configuration files and running the project.

---

## 1. CONFIG.TXT

Ensure that the following paths are correctly set in your `CONFIG.TXT` file:

/path/to/data/folder  
/path/to/clrernet/dir  
/path/to/hybridnets/dir  
/path/to/twinlitenet/dir  

---

## 2. SHADOW_ATTRS.csv

The `SHADOW_ATTRS.csv` defines the attributes of shadows. Below are the parameters:

- `width`: Width (in meters) of the negative shadow (int)
- `length`: Length (in meters) of the negative shadow (int)
- `beta`: Rotation (in degrees) of the positive shadow around the center (float)
- `transparency`: Transparency (range: 0-255) of the positive shadow (int)
- `blur`: Degree of blur (softness) of the positive shadow (int)
- `distance`: Distance (in meters) from the center lane to place the shadow (int)

### Editing Parameters

You can edit parameter bounds in `data/attrs_files/build_csv.py`. After making the changes, run the file to apply updates.

Set the number of CSV files in the `NUM_CSV` variable inside `masterscript.py`.

---

## Running the Project

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
