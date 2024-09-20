# 📝 Project Configuration Guide

Follow the steps below to ensure proper setup and configuration for the project.

---

## 🔧 Step 1: CONFIG.TXT Setup

Ensure the following paths are correctly set in your `CONFIG.TXT` file:

- `/path/to/data/folder`
- `/path/to/clrernet/dir`
- `/path/to/hybridnets/dir`
- `/path/to/twinlitenet/dir`

---

## 📊 Step 2: SHADOW_ATTRS.csv Setup

The `SHADOW_ATTRS.csv` file defines the attributes of shadows. Each parameter is described below:

- **width**: Width (in meters) of the negative shadow `(int)`
- **length**: Length (in meters) of the negative shadow `(int)`
- **beta**: Rotation (in degrees) of the positive shadow `(float)`
- **transparency**: Transparency (range: 0-255) of the positive shadow `(int)`
- **blur**: Degree of blur (softness) of the positive shadow `(int)`
- **distance**: Distance (in meters) from the center lane to place the shadow `(int)`

---

### 🔄 Editing Parameter Bounds

- Edit parameter bounds in the `data/attrs_files/build_csv.py` file.
- After editing, run the script to update the CSV files.
- Set the number of CSV files by updating the `NUM_CSV` variable in `masterscript.py`.

---

## 🚀 Step 3: Running the Project

1. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
