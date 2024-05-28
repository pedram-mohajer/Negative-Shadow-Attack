1. CONFIG.TXT
/path/to/data/folder
/path/to/clrernet/dir
/path/to/hybridnets/dir
/path/to/twinlitenet/dir

2. SHADOW_ATTRS.csv
width - width (int, meters) of negative shadow
length - length (int, meters) of negative shadow 
beta - rotation (float, degrees) of positive shadow around center
transparency - transparency (int [0-255]) of positive shadow
blur - degree of blur (int, softness) of positive shadow
distance - distance (int, meters) from center lane on which to place shadow

Edit parameter bounds in data/attrs_files/build_csv.py and run file.
Enter number of CSV files in NUM_CSV in masterscript.py

pip install -r requirements.txt
python3 masterscript.py
