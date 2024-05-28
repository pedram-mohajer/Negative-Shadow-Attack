#########################################
import os
import glob
import shutil
import cv2

def create_folders(config: dict) -> None:
    """ Ensures the existence of all parent folders and updates config

    Args:
        config: dict containing paths to all root folders
    """
    print("Ensuring all parent folders exist...")
    os.makedirs(config['input_folder'], exist_ok=True)
    os.makedirs(config['output_folder'], exist_ok=True)
    os.makedirs(config['results_folder'], exist_ok=True)
    
    data_dir = config['data_dir']
    clrernet_dir = config['clrernet_dir']
    hybridnets_dir = config['hybridnets_dir']
    twinlitenet_dir = config['twinlitenet_dir']

    clrernet_input = os.path.join(clrernet_dir, 'data/input')
    hybridnets_input = os.path.join(hybridnets_dir, 'data/input')
    twinlitenet_input = os.path.join(twinlitenet_dir, 'data/input')
    os.makedirs(clrernet_input, exist_ok=True)
    os.makedirs(hybridnets_input, exist_ok=True)
    os.makedirs(twinlitenet_input, exist_ok=True)
    config['clrernet_input'] = clrernet_input
    config['hybridnets_input'] = hybridnets_input
    config['twinlitenet_input'] = twinlitenet_input

    clrernet_output = os.path.join(clrernet_dir, 'data/output')
    hybridnets_output = os.path.join(hybridnets_dir, 'data/output')
    twinlitenet_output = os.path.join(twinlitenet_dir, 'data/output')
    os.makedirs(clrernet_output, exist_ok=True)
    os.makedirs(hybridnets_output, exist_ok=True)
    os.makedirs(twinlitenet_output, exist_ok=True)
    config['clrernet_output'] = clrernet_output
    config['hybridnets_output'] = hybridnets_output
    config['twinlitenet_output'] = twinlitenet_output

    evaluation_folder = os.path.join(data_dir, 'evaluation')
    os.makedirs(evaluation_folder, exist_ok=True)
    config['evaluation_folder'] = evaluation_folder

    clrernet_evaluation = os.path.join(evaluation_folder, 'clrernet')
    hybridnets_evaluation = os.path.join(evaluation_folder, 'hybridnets')
    twinlitenet_evaluation = os.path.join(evaluation_folder, 'twinlitenet')
    os.makedirs(clrernet_evaluation, exist_ok=True)
    os.makedirs(hybridnets_evaluation, exist_ok=True)
    os.makedirs(twinlitenet_evaluation,  exist_ok=True)
    config['clrernet_evaluation'] = clrernet_evaluation
    config['hybridnets_evaluation'] = hybridnets_evaluation
    config['twinlitenet_evaluation'] = twinlitenet_evaluation

    overlaid_imgs = os.path.join(data_dir, 'output/overlaid_imgs')
    os.makedirs(overlaid_imgs, exist_ok=True)
    config['overlaid_imgs'] = overlaid_imgs

def delete_and_make(folder_list: list) -> None:
    """ Deletes and recreates all folders in folder_list

    Args:
        folder_list: list of folder paths to delete/recreate
    """
    for folder in folder_list:
        print(f'Deleting and recreating {folder}')
        shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

def clear_outputs(config: dict) -> None:
    """ Clears all output folders of images, with the unfortunate consequence
    of deleting the parent folder as well.
    
    Args:
        config: dict containing all parent folder paths
    """
    outputs = [config[folder] for folder in config if 'output' in config[folder] and folder != 'output_folder']
    inputs = [config[folder] for folder in config if 'input' in config[folder] and folder != 'input_folder']
    evaluation = [config[folder] for folder in config if 'evaluation' in config[folder] and folder != 'evaluation_folder']
    
    delete_and_make(inputs)
    delete_and_make(outputs)
    delete_and_make(evaluation)

    
def resize_images(imgs_dir: os.path) -> None:
    """ Resizes all the images in the chosen directory to 1640x590

    Args:
        output_dir: path to the directory containing the images to resize
    """
    IMG_X = 1640
    IMG_Y = 590

    for subdir, _, files in os.walk(imgs_dir):
        if files:
            print(f'Resizing all files in {subdir} to {IMG_X}x{IMG_Y}...')

            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    orig_file_path = os.path.join(subdir, file)
                    img = cv2.imread(orig_file_path)
                    resized_img = cv2.resize(img, (IMG_X, IMG_Y))
                    cv2.imwrite(orig_file_path, resized_img)

def move2algos(config: dict) -> None:
    """ Copies all images from all folders in a given directory to the data/input folder of each algortithm directory

    Args:
        config: dict containing all parent folders
    """
    print('Moving all input image files to LD input folders...')
    input_folder = config['input_folder']
    clrernet_input = config['clrernet_input']
    hybridnets_input = config['hybridnets_input']
    twinlitenet_input = config['twinlitenet_input']

    shutil.copytree(input_folder, clrernet_input, dirs_exist_ok=True)
    shutil.copytree(input_folder, hybridnets_input, dirs_exist_ok=True)
    shutil.copytree(input_folder, twinlitenet_input, dirs_exist_ok=True)
    
    print('Moving all overlaid image files to LD input folders...')
    overlaid_folder = config['overlaid_imgs']
    shutil.copytree(overlaid_folder, clrernet_input, dirs_exist_ok=True)
    shutil.copytree(overlaid_folder, hybridnets_input, dirs_exist_ok=True)
    shutil.copytree(overlaid_folder, twinlitenet_input, dirs_exist_ok=True)


def move2root(config: dict) -> None:
    """ Copies all images from the data/output to the masterscript data directory

    Args:
        config: dict containing all parent folders
    """
    print('Moving all output image files from LD output folders to main evaluation folder')
    clrernet_output = config['clrernet_output']
    hybridnets_output = config['hybridnets_output']
    twinlitenet_output = config['twinlitenet_output']

    clrernet_evaluation = config['clrernet_evaluation']
    hybridnets_evaluation = config['hybridnets_evaluation']
    twinlitenet_evaluation = config['twinlitenet_evaluation']

    shutil.copytree(clrernet_output, clrernet_evaluation, dirs_exist_ok=True)
    shutil.copytree(hybridnets_output, hybridnets_evaluation, dirs_exist_ok=True)
    shutil.copytree(twinlitenet_output, twinlitenet_evaluation, dirs_exist_ok=True)
