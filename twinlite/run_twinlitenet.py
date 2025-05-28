import os
import shutil

import cv2
import torch
import numpy as np
from tqdm.autonotebook import tqdm

from model import TwinLite as net

def Run(model, img):
    # Resize and keep a copy for drawing
    img = cv2.resize(img, (640, 360))
    img_rs = img.copy()

    # Prepare tensor [1×3×360×640], RGB, normalized
    img = img[:, :, ::-1].transpose(2, 0, 1)          # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
    img = img.cuda()

    # Inference
    with torch.no_grad():
        img_out = model(img)
    x0, x1 = img_out

    # Argmax over channel dim to get per-pixel class
    _, da_predict = torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)

    # Convert to CPU numpy masks
    DA = da_predict.byte().cpu().numpy()[0] * 255
    LL = ll_predict.byte().cpu().numpy()[0] * 255

    # Count connected components in lane mask (subtract 1 for background)
    num_labels, labels_im = cv2.connectedComponents(LL.astype(np.uint8))
    lane_count = num_labels - 1

    # Overlay lane mask in green
    img_rs[LL > 100] = [0, 255, 0]

    return img_rs, lane_count


if __name__ == "__main__":
    # Load the model
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('pretrained/best.pth'))
    model.eval()

    # Prepare input/output directories
    image_dir = 'images'
    result_dir = 'results'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    # Process each image
    image_list = os.listdir(image_dir)
    for img_name in tqdm(image_list, desc="Processing images"):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not load {img_name}")
            continue

        # Run inference and get lane count
        result_img, lane_count = Run(model, img)

        # Print the filename and detected lane count
        print(f"{img_name}: {lane_count}")

        # Save the overlay image
        out_path = os.path.join(result_dir, img_name)
        cv2.imwrite(out_path, result_img)
