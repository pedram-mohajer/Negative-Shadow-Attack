import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
ROI_H = 302
IMAGE_W = 1640
IMAGE_H = 590-ROI_H

LANE_MARK_POS_X = 807.5
LANE_MARK_WIDTH_PIX = 1
RIGHT_LANE_WIDTH_M = 4
RIGHT_LANE_WIDTH_PIX = 835.5 - LANE_MARK_POS_X + LANE_MARK_WIDTH_PIX
BEV_LEFT_ROAD_EDGE_POS = 777.5
BEV_LANE_WIDTH_PIX = 835.5 - BEV_LEFT_ROAD_EDGE_POS

PIX_TO_M = RIGHT_LANE_WIDTH_M / RIGHT_LANE_WIDTH_PIX 
M_TO_PIX = RIGHT_LANE_WIDTH_PIX / RIGHT_LANE_WIDTH_M

src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32([[800, IMAGE_H], [840, IMAGE_H], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst)
M_INV = cv2.getPerspectiveTransform(dst, src)

def bev_tranfom(img_path: os.path) -> tuple[np.ndarray, np.ndarray]:
    """ Transforms an image from 3d to 2d
    
    Args:
        img_path: path to image to transform
    
    Returns:
        image pre transform and image post transform
    """
    og_img = cv2.imread(img_path)
    img = og_img[ROI_H: (ROI_H + IMAGE_H), 0:IMAGE_W]
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H), flags=cv2.INTER_NEAREST)

    return og_img, warped_img

def inv_bev_transform(og_img: np.ndarray, post_overlay_img: np.ndarray) -> np.ndarray:
    """ Transforms an image from 3d back to 2d

    Args:
        og_image: original image
        post_overlay_img: 2d image with overlaid shadow
    
    Returns:
        3d image with overlay
    """
    inv_bev_img = cv2.warpPerspective(post_overlay_img, M_INV, (IMAGE_W, IMAGE_H), flags=cv2.INTER_NEAREST)
    og_img[ROI_H: (ROI_H + IMAGE_H), 0:IMAGE_W] = inv_bev_img

    return og_img