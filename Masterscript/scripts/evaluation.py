#########################################
import os
import cv2
import datetime
import numpy as np
import shutil
from scripts.results_logging import create_workbook, write_results, save_workbook

def build_pre_dict(data_dir: os.path) -> dict:
    """ Builds a dict containing all images with an attack but before lane detection

    Args:
        data_dir: path to folder containing pre-LD/post-attack images

    Returns:
        dict containing all images without lane detection but post attack
    """
    preld_dict = {}

    for subdir, _, files in os.walk(data_dir):
        if files:
            parent_dir = os.path.basename(os.path.dirname(subdir))
            current_dir = os.path.basename(subdir)
            if parent_dir not in preld_dict:
                preld_dict[parent_dir] = {}

            if current_dir not in preld_dict[parent_dir]:
                preld_dict[parent_dir][current_dir] = {}

            for file in files:
                img_path = os.path.join(subdir, file)
                preld_dict[parent_dir][current_dir][file] = cv2.imread(img_path)

    return preld_dict
            
def build_image_dicts(data_dir: os.path) -> (dict, dict):
    """ Builds a dict containing all images without an attack
    
    Args:
        data_dir: path to folder containing all evaluation images
    
    Returns:
        control images dict, and attack images dict

        contains all images for the respective dict types
        
        dict ={
            subfolder = {
                image_name = np.ndarray
            }
        }
    """
    control_images = {}
    attack_images = {}
    for subdir, _, files in os.walk(data_dir):
        basename = os.path.basename(subdir)

        if 'overlaid' not in subdir:
            control_images[basename] = {}

            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    img = os.path.join(subdir, file)
                    control_images[basename][file] = cv2.imread(img)

        else:
            attack_images[basename] = {}

            for attack_dir, _, attack_files in os.walk(subdir):
                attack_basename = os.path.basename(attack_dir)
                if attack_basename != basename:
                    attack_images[basename][attack_basename] = {}

                    for attack_file in attack_files:
                        if attack_file.endswith('.png') or file.endswith('.jpg'):
                            attack_img = os.path.join(attack_dir, attack_file)
                            attack_images[basename][attack_basename][attack_file] =  cv2.imread(attack_img)

    return control_images, attack_images

def evaluate_images(attack_dict, control_dict, preld_dict, pre_atk_dict, options):
    """ Performs attack evaluation on images given the options specified
    
    Args:
        attack_dict: dictionary of images containging post-attack images
        control_dict: dictionary of images containing pre-attack images
        preld_dict: dictionary of images containing post-attack but pre lane detection images
        pre_atk_dict: dictionary of images pre attack
        options: options for evaluations
    
    Returns:
        dict containing results for every attack image
    """
    CROP_H, CROP_W_L, CROP_W_R, MAX_INTENSITY, BLOCK_SIZE, C, CUTOFF = options
    results = {}
    for control_folder in control_dict:
        attack_name = control_folder + '_overlaid'

        if attack_name in attack_dict:
            attack_folder = attack_dict[attack_name]

            results[attack_name] = {}
            for attack_type in attack_folder:
                for attack_image in attack_dict[attack_name][attack_type]:
                    if not attack_image in results[attack_name]:
                        results[attack_name][attack_image] = {}

                    attack = attack_dict[attack_name][attack_type][attack_image]
                    control = control_dict[control_folder][attack_image]
                    preld = preld_dict[attack_name][attack_type][attack_image]
                    pre_atk = pre_atk_dict[control_folder][attack_image]

                    x,y,_ = pre_atk.shape
                    a,b,_ = attack.shape
                    if (x != a) or (y != b):
                        pre_atk = cv2.resize(pre_atk, (b,a))
                        preld = cv2.resize(preld, (b, a))

                    # Crop images to account for BEV artifacts
                    attack = attack[:CROP_H, CROP_W_L:CROP_W_R]
                    control = control[:CROP_H, CROP_W_L:CROP_W_R]
                    preld = preld[:CROP_H, CROP_W_L:CROP_W_R]
                    pre_atk = pre_atk[:CROP_H, CROP_W_L:CROP_W_R]

                    # lanes detected pre_attack
                    diff_pre_atk = cv2.subtract(pre_atk, control)
                    # lanes detected post attack
                    diff_pst_atk = cv2.subtract(preld, attack)

                    # lanes are added
                    diff_ac = cv2.subtract(diff_pst_atk, diff_pre_atk)
                    # lanes are removed
                    diff_ca = cv2.subtract(diff_pre_atk, diff_pst_atk)


                    # Convert to images to greyscale, applies addaptive greyscale thresholding
                    # to account for BEV artifacts and detect lanes
                    diff_ac = cv2.cvtColor(diff_ac, cv2.COLOR_BGR2GRAY)
                    diff_ca = cv2.cvtColor(diff_ca, cv2.COLOR_BGR2GRAY)

                    thresh_ac = cv2.adaptiveThreshold(diff_ac, MAX_INTENSITY, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                          cv2.THRESH_BINARY_INV, BLOCK_SIZE, C)
                    thresh_ca = cv2.adaptiveThreshold(diff_ca, MAX_INTENSITY, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                         cv2.THRESH_BINARY_INV, BLOCK_SIZE, C)
                    
                    pix_ac = cv2.countNonZero(thresh_ac)
                    pix_ca = cv2.countNonZero(thresh_ca)

                    if (pix_ac >= CUTOFF):
                        # Lanes were added to control
                        img_results = (True, 1)

                        if (pix_ca >= CUTOFF):
                            # Lanes were added and removed
                            img_results = (True, 2)

                    elif (pix_ca >= CUTOFF):
                        # Lanes were removed
                        img_results = (True, -1)
                    
                    else:
                        # No change, attack was unsuccessful
                        img_results = (False, 0)

                    results[attack_name][attack_image].update({attack_type:img_results})

    return results


def evaluate_clrernet(clrernet_data_dir: os.path, preld_imgs: dict, pre_atk_imgs: dict) -> None:
    """ Evaluates attack succcess on the clrernet output files

    Args:
        clrernet_data_dir: path to folder containg clrernet data
        preld_imgs: dictionary of all images pre lane detection
        pre_atk: dictionary of all images pre attack
    """
    CROP_H = 540
    CROP_W_L = 20
    CROP_W_R = 1600

    MAX_INTENSITY = 255 # Maximum pixel intensity
    BLOCK_SIZE = 31 # Size of a pixel neighborhood
    C = 12 # Constant subtracted from mean

    # Tests said three added lanes amounted to 9235 pix
    # Rounded down and added room for error, divided by 3
    CUTOFF = 2500
    options = (CROP_H, CROP_W_L, CROP_W_R, MAX_INTENSITY, BLOCK_SIZE, C, CUTOFF)
    print('Evaluating CLRerNets images...')
    control_dict, attack_dict = build_image_dicts(clrernet_data_dir)
    results = evaluate_images(attack_dict, control_dict, preld_imgs, pre_atk_imgs, options)

    return results

                    
def evaluate_hybridnets(hybridnets_data_dir: os.path, preld_imgs: dict, pre_atk_imgs: dict) -> None:
    """ Evaluates attack success on the hybridnets output files

    Args:
        hybridnets_data_dir: path to folder containng hybridnets data
        preld_imgs: dictionary of all images pre lane detection
        pre_atk: dictionary of all images pre attack
    """
    CROP_H = 540
    CROP_W_L = 20
    CROP_W_R = 1600

    MAX_INTENSITY = 255 # Maximum pixel intensity
    BLOCK_SIZE = 31 # Size of a pixel neighborhood
    C = 12 # Constant subtracted from mean

    # Tests show an added lane is 1826 pixels
    # Rounding down for error
    CUTOFF = 8000
    options = (CROP_H, CROP_W_L, CROP_W_R, MAX_INTENSITY, BLOCK_SIZE, C, CUTOFF)
    print('Evaluating Hybridnets images...')
    control_dict, attack_dict = build_image_dicts(hybridnets_data_dir)
    results = evaluate_images(attack_dict, control_dict, preld_imgs, pre_atk_imgs, options)
    
    return results

def evaluate_twinlitenet(twinlitenet_data_dir: os.path, preld_imgs: dict, pre_atk_imgs: dict) -> None:
    """ Evaluates attack success on twinlitenet output files

    Args:
        twinltitenets_data_dir: path to folder containing twinlitenet data
        preld_imgs: dictionary of all images pre lane detection
        pre_atk: dictionary of all images pre attack
    """
    CROP_H = 355
    CROP_W_L = 10
    CROP_W_R = 630

    MAX_INTENSITY = 255 # Maximum pixel intensity
    BLOCK_SIZE = 21 # Size of a pixel neighborhood
    C = 12 # Constant subtracted from mean

    # One added lane was observed to be 2780 through  testing
    # Rounding down to account for possible error
    CUTOFF = 1800
    options = (CROP_H, CROP_W_L, CROP_W_R, MAX_INTENSITY, BLOCK_SIZE, C, CUTOFF)
    print('Evaluating TwinLiteNet images...')
    control_dict, attack_dict = build_image_dicts(twinlitenet_data_dir)
    results = evaluate_images(attack_dict, control_dict, preld_imgs, pre_atk_imgs, options)

    return results

def evaluate_attacks(config: dict, csv_num: int) -> None:
    """ Evaluates the attack of all LD algorithms by comparing the number of lane pixels in each image

    Args:
        data_dir: path to folder containg the data to evaluate
        csv_num: current csv file number
    """ 
    results_workbook = create_workbook()
    data_dir = config['data_dir']
    clrernet_data_dir = os.path.join(data_dir, 'evaluation/clrernet')
    hybridnets_data_dir = os.path.join(data_dir, 'evaluation/hybridnets')
    twinlitenet_data_dir = os.path.join(data_dir, 'evaluation/twinlitenet')
    preld_data_dir = os.path.join(data_dir, 'output/overlaid_imgs')
    pre_atk_dir = os.path.join(data_dir, 'input/')

    preld_imgs = build_pre_dict(preld_data_dir)
    print('pre attack dicts built')    

    pre_atk_imgs = build_pre_dict(pre_atk_dir)['input']
    print('pre and post attack dicts built')    

    clrernet_results = evaluate_clrernet(clrernet_data_dir, preld_imgs, pre_atk_imgs)
    hybridnets_results = evaluate_hybridnets(hybridnets_data_dir, preld_imgs, pre_atk_imgs)
    twinlitenet_results = evaluate_twinlitenet(twinlitenet_data_dir, preld_imgs, pre_atk_imgs)
    
    write_results(results_workbook, 'clrernet_results', clrernet_results)
    write_results(results_workbook, 'hybridnets', hybridnets_results)
    write_results(results_workbook, 'twinlitenet', twinlitenet_results)

    results_folder = os.path.join(config['results_folder'], f'attr_{csv_num}')
    os.makedirs(results_folder, exist_ok=True)

    file_time = datetime.datetime.now().strftime("%d_%m_%y, %H:%M")
    results_path = os.path.join(results_folder, f'Results{csv_num} {file_time} .xlsx')

    save_workbook(results_workbook, results_path)
    
    old_csv_path = os.path.join(config['attr_folder'], f'shadow_attrs{csv_num}.csv')
    new_csv_path = os.path.join(results_folder, f'shadow_attrs{csv_num}.csv')
    shutil.copy(old_csv_path, new_csv_path)
