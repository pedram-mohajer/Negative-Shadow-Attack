#########################################
import cv2, csv, os
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scripts.bev_transform import *
import matplotlib.pyplot as plt

TRANSPARENCY = 90 # Alpha blending between the negative and positive shadows
BLUR = 0 # Blur of the edges of the negative shadows
CURB_K = 10 # Extra length needed to ensure positive shadow goes over the curb

def get_output_folder(basename: str, output_path: os.path) -> os.path:
    """ Creates output folder based on the basename
    Does not create a subfolder if the basename is input as input is not a subfolder

    Args:
        basename: basename of subdirectory
        output_path: root path of output folder
    
    Returns:
        path of output folder for subdirectory
    """
    if basename != 'input':
        output_dir = os.path.join(output_path, basename)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    os.makedirs(output_path, exist_ok=True)
    return output_path

def cv2_to_PIL(cv2_img: np.ndarray) -> Image.Image:
    """ Converts OpenCV image to the Pillow Image format

    Args:
        cv2_img: OpenCv style image to convert
    
    Returns:
        Original image in the PIL Image format
    """
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGRA2RGBA))

def PIL_to_cv2(PIL_img: Image.Image) -> np.ndarray:
    """ Converts Pillow Image format to an OpenCV image object

    Args:
        PIL_img: PIL image to convert
    
    Returns:
        OpenCV image object created from passed image
    """
    return cv2.cvtColor(np.asarray(PIL_img), cv2.COLOR_RGB2BGR)

def get_center_coords(overlay_size: tuple[int, int], img_size: tuple[int, int]) -> tuple[int, int]:
    """ Generates the X/Y coordinates necessary for applying overlays as the origin for a PIL object is at the top left

    Args:
        overlay_size: width, height of overlay
        img_size: width, height of image
    
    Returns:
        Centered coordinates
    """
    midx = img_size[0]//2
    midy = img_size[1]//2

    adjustedx = midx - overlay_size[0]//2
    adjustedy = midy - overlay_size[1]//2

    return tuple((adjustedx, adjustedy))

def paste_overlay(row: dict, warped_img: np.ndarray) -> np.ndarray:
    """ Applies specified shadow attributes to the passed in BEV transformed image

    Args:
        row: dict containing all attributes for a shadow attack
        warped_img: BEV transformed image on which to apply the shadow
    
    Returns:
        BEV transformed image with a pasted shadow attack
    """
    width = row['width']
    length = row['length']
    beta = row['beta']
    distance = row['distance']
    transparency = row['transparency']
    blur = row['blur']
    road_width_px = round(BEV_LANE_WIDTH_PIX) + CURB_K
    warped_PIL = cv2_to_PIL(warped_img)

    width = round(width * M_TO_PIX)
    length = int(length * M_TO_PIX)
    distance = int(distance * M_TO_PIX)
    positive_shadow= Image.new('RGBA', (road_width_px, length),  (0,0,0, transparency))

    negative_rotation = -1 * beta
    negative_shadow = Image.new('RGBA', (width, length), (0, 0, 0, 0))
    negative_shadow = negative_shadow.rotate(negative_rotation, expand=True, fillcolor=(0, 0, 0, transparency))

    center = get_center_coords(negative_shadow.size, positive_shadow.size)
    offset = ((road_width_px//2) + distance, center[1])
    completed_overlay = positive_shadow.copy()
    completed_overlay.paste(negative_shadow, offset)
    
    blured_overlay = ImageOps.expand(completed_overlay, border=blur, fill=(0,0,0,0))
    completed_overlay = blured_overlay.filter(ImageFilter.BoxBlur(blur))

    
    final_PIL = warped_PIL.copy()
    center = get_center_coords(completed_overlay.size, final_PIL.size)
    offset = (round(BEV_LEFT_ROAD_EDGE_POS)-CURB_K//2+1, 275-length)
    final_PIL.alpha_composite(completed_overlay, offset)

    return PIL_to_cv2(final_PIL)


def apply_overlay(config: dict, csv_num: int, verbose=False) -> None:
    """ Applies every shadow attack specified in shadow_attrs.csv to every image in the data directory

    Args:
        data_dir: path to the root directory of the data folder
        csv_num: number of csv file to read
    """
    wrapper = list()
    headers = list()
    attr_filename = os.path.join(config['attr_folder'], f'shadow_attrs{csv_num}.csv')
    attr_file = open(attr_filename, newline='')
    reader = csv.reader(attr_file)

    for i, row in  enumerate(reader):
        if i == 0:
            headers = row
        else:
            wrapper.append({key:eval(value) for key, value in zip(headers,row)})

    input_folder = config['input_folder']
    output_folder = config['overlaid_imgs']
    
    i = 0
    print(f'Applying shadow file {csv_num} to images')
    for row in wrapper:
        for subdir,_, files in os.walk(input_folder):
            if files:
                if verbose:
                    print(f'Applying shadow num {i} to files in {subdir}')
                output_dir = get_output_folder(os.path.basename(subdir), output_folder)+'_overlaid'
                os.makedirs(output_dir, exist_ok=True)
                
                row_dir = os.path.join(output_dir, f'{i}')
                os.makedirs(row_dir, exist_ok=True)

                for file in files:
                    if file.endswith('.png') or file.endswith('.jpg'):
                        img_path = os.path.join(subdir, file)
                        og_img, warped_img = bev_tranfom(img_path)
                        overlay_bev = paste_overlay(row, warped_img)

                        final_img = inv_bev_transform(og_img, overlay_bev)
                        final_img_path = os.path.join(row_dir, file)
                        cv2.imwrite(final_img_path, final_img)
        i+=1