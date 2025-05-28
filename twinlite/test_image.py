import torch
import numpy as np
import shutil
import os
import cv2
import pandas as pd
import ast
from model import TwinLite as net

def group_lane_pixels(LL_mask):
    num_labels, labels = cv2.connectedComponents(LL_mask)
    groups = []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        coords = list(zip(xs.tolist(), ys.tolist()))
        groups.append(coords)
    return groups

def Run(model, img):
    img = cv2.resize(img, (640, 360))
    img_rs = img.copy()

    # Fix negative stride issue
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0).cuda().float() / 255.0

    with torch.no_grad():
        x0, x1 = model(img)

    _, ll_predict = torch.max(x1, 1)
    LL = ll_predict.byte().cpu().numpy()[0] * 255
    img_rs[LL > 100] = [0, 255, 0]
    pixel_groups = group_lane_pixels(LL)
    return img_rs, pixel_groups


# Load shadow data
csv_path = "NS_Images_Driver/shadow_lengths_driver.csv"
df_shadow = pd.read_csv(csv_path)
df_shadow.set_index("Image_Name", inplace=True)

# Load model
model = net.TwinLiteNet()
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('pretrained/best.pth'))
model.eval()

# Prepare output folder
if os.path.exists('results'):
    shutil.rmtree('results')
os.mkdir('results')

# Run inference
image_list = sorted(os.listdir('images'))
for imgName in image_list:
    img = cv2.imread(os.path.join('images', imgName))
    if img is None:
        continue

    result_img, pixel_groups = Run(model, img)
    cv2.imwrite(os.path.join('results', imgName), result_img)

    # Flatten all predicted lane pixels
    predicted_pixels = set([pt for group in pixel_groups for pt in group])

    # Find corresponding bright area pixels
    base_name = os.path.splitext(imgName)[0]
    #driver_csv_name = f"{base_name}.png"  # match name with CSV
    driver_csv_name = base_name.replace("_driver", "") + ".png"

    if driver_csv_name not in df_shadow.index:
        print(f"⚠️ No bright region info for {driver_csv_name}")
        continue

    try:
        bright_pixels_str = df_shadow.loc[driver_csv_name, "BrightPixels_Driver"]
        #bright_pixels = set(tuple(map(int, pt)) for pt in ast.literal_eval(bright_pixels_str))
        # Parse and downscale to match model's 640×360 resolution
        bright_pixels = set(
            (int(x * 0.5), int(y * 0.5)) for (x, y) in ast.literal_eval(bright_pixels_str)
        )

        intersection = predicted_pixels & bright_pixels
        overlap_ratio = len(intersection) / len(bright_pixels) if len(bright_pixels) > 0 else 0

        if overlap_ratio > 0.10:
            print(f"\n🖼 Image: {imgName}")
            print(f"→ Overlap with bright area: {len(intersection)} pixels ({overlap_ratio:.2%})")
            print("✅ Bright area WAS detected as lane (over 10% overlap).")



    except Exception as e:
        print(f"⚠️ Error processing bright pixels for {imgName}: {e}")



# import torch
# import numpy as np
# import shutil
# import os
# import cv2
# from model import TwinLite as net

# def group_lane_pixels(LL_mask):
#     # Perform connected component analysis
#     num_labels, labels = cv2.connectedComponents(LL_mask)
#     groups = []
#     for label in range(1, num_labels):  # skip background label 0
#         ys, xs = np.where(labels == label)
#         coords = list(zip(xs.tolist(), ys.tolist()))
#         groups.append(coords)
#     return groups

# def Run(model, img):
#     img = cv2.resize(img, (640, 360))
#     img_rs = img.copy()

#     img = img[:, :, ::-1].transpose(2, 0, 1)
#     img = np.ascontiguousarray(img)
#     img = torch.from_numpy(img)
#     img = torch.unsqueeze(img, 0).cuda().float() / 255.0

#     with torch.no_grad():
#         img_out = model(img)
#     x0 = img_out[0]
#     x1 = img_out[1]

#     _, da_predict = torch.max(x0, 1)
#     _, ll_predict = torch.max(x1, 1)

#     DA = da_predict.byte().cpu().numpy()[0] * 255
#     LL = ll_predict.byte().cpu().numpy()[0] * 255

#     img_rs[LL > 100] = [0, 255, 0]

#     lane_groups = group_lane_pixels(LL)
#     return img_rs, lane_groups

# # Load model
# model = net.TwinLiteNet()
# model = torch.nn.DataParallel(model).cuda()
# model.load_state_dict(torch.load('pretrained/best.pth'))
# model.eval()

# # Prepare output folder
# if os.path.exists('results'):
#     shutil.rmtree('results')
# os.mkdir('results')

# # Run inference
# image_list = sorted(os.listdir('images'))
# for imgName in image_list:
#     img = cv2.imread(os.path.join('images', imgName))
#     if img is None:
#         continue

#     result_img, pixel_groups = Run(model, img)
#     cv2.imwrite(os.path.join('results', imgName), result_img)

#     print(f"\n🖼 Image: {imgName}")
#     for i, group in enumerate(pixel_groups):
#         print(f"  Lane {i+1} — {len(group)} pixels")
#         print(f"    {group[:10]}{' ...' if len(group) > 10 else ''}")  # show first 10
