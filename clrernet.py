# clrernet.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from argparse import ArgumentParser
import cv2
import os
import numpy as np
from typing import List, Tuple
import sys

sys.path.insert(0, os.path.abspath('clrer'))

from mmdet.apis import init_detector
from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes



def resize_to_clrernet_format(image_path: str, width: int = 1640, height: int = 590):
    """
    Resizes an image to the default CLRerNet input size and overwrites it.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(image_path, resized_img)
    #print(f"✅ Resized and saved: {image_path}")


def run_clrernet(model, bright_pixels_driver: List[Tuple[int, int]], shadow_id: int) -> Tuple[bool, float]:
    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3)
    args = parser.parse_args([])  # Allow calling from script

    # Prepare image
    img_name = f"sun_cast_{shadow_id}.png"
    img_path = os.path.join("NS_Images_Driver", img_name)
    resize_to_clrernet_format(img_path)

    # Load model and run inference

    model = init_detector(
        "clrer/configs/clrernet/culane/clrernet_culane_dla34_ema.py",
        "clrer/configs/clrernet_culane_dla34_ema.pth",
        device=args.device
    )

    src, preds = inference_one_image(model, img_path)

    # Flatten all predicted lane points
    flat_preds = [pt for lane in preds for pt in lane]
    lane_px = set((int(round(x)), int(round(y))) for (x, y) in flat_preds)  # (x, y)

    # Scale bright pixels to match resized image
    scale_x = 1640 / 1280
    scale_y = 590 / 720
    bright_px = set(
        (int(round(x * scale_x)), int(round(y * scale_y)))
        for (x, y) in bright_pixels_driver
    )

    # Overlap check
    inter = lane_px & bright_px
    overlap = len(inter) / len(bright_px) if bright_px else 0


    category = "Detected" if overlap > 0.01 else "Undetected"
    #print(f"[CLRerNet] shadow_id={shadow_id} | bright:{len(bright_px)} lane:{len(lane_px)} " f"→ inter:{len(inter)} ({overlap:.2%}) → {category}")



    # Save output
    os.makedirs("results/CLRerNet/Detected", exist_ok=True)
    os.makedirs("results/CLRerNet/Undetected", exist_ok=True)
    save_path = os.path.join("results", "CLRerNet", category, img_name)

    dst = visualize_lanes(src, preds)  # overlay lanes
    cv2.imwrite(save_path, cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))


    return category == "Detected", overlap


#run_clrernet( [(1,1),(2,2)] , shadow_id=1)