# based on https://github.com/open-mmlab/mmdetection/blob/v2.28.0/demo/image_demo.py
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import init_detector
import cv2

from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes
import numpy as np

from typing import List, Tuple

import os



def resize_to_clrernet_format(image_path: str, width: int = 1640, height: int = 590):
    """
    Resizes an image to the default CLRerNet input size (1640x590) and overwrites it.

    Args:
        image_path (str): Path to the image to resize.
        width (int): Target width (default = 1640).
        height (int): Target height (default = 590).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(image_path, resized_img)
    print(f"✅ Resized and saved: {image_path}")


def run_clrernet(bright_pixels_driver: List[Tuple[int, int]], shadow_id: int):

    parser = ArgumentParser()
    parser.add_argument('--device',   default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    # build the model from a config file and a checkpoint file

    img_name = f"sun_cast_{shadow_id}.png"
    img_path = os.path.join("NS_Images_Driver", img_name)

    resize_to_clrernet_format(img_path)

    model = init_detector("configs/clrernet/culane/clrernet_culane_dla34_ema.py", "clrernet_culane_dla34_ema.pth", device=args.device)
    # test a single image
    
    src, preds = inference_one_image(model, img_path) #input
    # show the results

    flat_preds = [pt for lane in preds for pt in lane]


    print("🔍 Sample lane pixels (flat_preds):", flat_preds[:20])
 


    dst = visualize_lanes(src, preds, save_path='result.png') #output saved



    # h, w = src.shape[:2]
    # lane_mask = np.zeros((h, w), dtype=np.uint8)
    # for lane_pts in preds:
    #     lane_pts = lane_pts.astype(np.int32)
    #     for p1, p2 in zip(lane_pts[:-1], lane_pts[1:]):
    #         cv2.line(lane_mask, tuple(p1), tuple(p2), 255, thickness=4)
    # cv2.imwrite('lane_mask.png', lane_mask)

pixels= [(0,0),(1,1)]
category = run_clrernet(pixels, shadow_id=4)
print(category)




