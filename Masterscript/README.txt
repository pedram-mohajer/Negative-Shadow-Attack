
# Project Configuration Guide

This guide will help you set up the necessary configuration files and parameters for the project.

---

## 1️⃣ CONFIG.TXT

Ensure the following paths are correctly set in your `CONFIG.TXT` file:

/path/to/data/folder  
/path/to/clrernet/dir  
/path/to/hybridnets/dir  
/path/to/twinlitenet/dir  

---

## 2️⃣ SHADOW_ATTRS.csv

The `SHADOW_ATTRS.csv` defines the attributes of shadows in the project. Here are the parameters:

| **Attribute**    | **Description**                                        | **Type**          |
|------------------|--------------------------------------------------------|-------------------|
| `width`          | Width (in meters) of the negative shadow               | `int`             |
| `length`         | Length (in meters) of the negative shadow              | `int`             |
| `beta`           | Rotation (in degrees) of the positive shadow           | `float`           |
| `transparency`   | Transparency (range: 0-255) of the positive shadow     | `int`             |
| `blur`           | Degree of blur (softness) of the positive shadow       | `int`             |
| `distance`       | Distance (in meters) from the center lane              | `int`             |

### ⚙️ Editing Parameter Bounds

- You can edit the bounds for these parameters in the file located at `data/attrs_files/build_csv.py`.
- After editing, run the script to update the CSV files.

### 🛠️ Updating Masterscript

- Specify the number of CSV files in the `NUM_CSV` variable within `masterscript.py`.

---

## 🚀 Running the Project

1. Install the required dependencies:

```bash
pip install -r requirements.txt
