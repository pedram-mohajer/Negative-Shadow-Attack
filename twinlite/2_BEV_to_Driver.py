import os
import cv2
import numpy as np
import pandas as pd
import ast

# Folders
bev_folder    = "NS_Images_BEV"
driver_folder = "NS_Images_Driver"
os.makedirs(driver_folder, exist_ok=True)

# Homography point correspondences
pts_bev = np.float32([
    [6,   486],
    [6,   266],
    [1276, 270],
    [1276, 488]
])
pts_driver = np.float32([
    [1120, 704],
    [7,    574],
    [588,  392],
    [674,  394]
])

# Compute BEV → driver homography
H_bev_to_driver, _ = cv2.findHomography(pts_bev, pts_driver)

# Load a driver-view template just to get output size
driver_template = cv2.imread("Driver.png")
if driver_template is None:
    raise FileNotFoundError("Cannot load Driver.png")
h_out, w_out = driver_template.shape[:2]

# Warp each image from BEV to driver view
for fname in os.listdir(bev_folder):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        continue

    bev_path = os.path.join(bev_folder, fname)
    bev_img  = cv2.imread(bev_path)
    if bev_img is None:
        print(f"⚠️ Skipping {fname}: failed to load")
        continue

    # Warp image
    warped = cv2.warpPerspective(bev_img, H_bev_to_driver, (w_out, h_out))

    # Output name
    base, ext = os.path.splitext(fname)
    out_fname = f"{base}_driver{ext}"
    out_path = os.path.join(driver_folder, out_fname)

    cv2.imwrite(out_path, warped)
    print(f"✅ Saved {out_fname}")

# === CSV BrightPixels transformation ===
csv_path = os.path.join(bev_folder, "shadow_lengths.csv")
df = pd.read_csv(csv_path)

def transform_pixel_list(pixel_str):
    try:
        pixels = ast.literal_eval(pixel_str)
        if not pixels:
            return []
        pts = np.array(pixels, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H_bev_to_driver).reshape(-1, 2)
        return [tuple(map(int, pt)) for pt in transformed]
    except Exception as e:
        return []

df["BrightPixels_Driver"] = df["BrightPixels"].apply(transform_pixel_list)

# Save updated CSV next to driver images
csv_out_path = os.path.join(driver_folder, "shadow_lengths_driver.csv")
df.to_csv(csv_out_path, index=False)
print(f"📄 CSV saved to {csv_out_path}")

print("✅ All done!")


# import os
# import cv2
# import numpy as np

# # Folders
# bev_folder    = "NS_Images_BEV"
# driver_folder = "NS_Images_Driver"
# os.makedirs(driver_folder, exist_ok=True)

# # Homography point correspondences
# pts_bev = np.float32([
#     [6,   486],
#     [6,   266],
#     [1276, 270],
#     [1276, 488]
# ])
# pts_driver = np.float32([
#     [1120, 704],
#     [7,    574],
#     [588,  392],
#     [674,  394]
# ])

# # Compute BEV → driver homography once
# H_bev_to_driver, _ = cv2.findHomography(pts_bev, pts_driver)

# # Load a driver-view template just to get the output size
# driver_template = cv2.imread("Driver.png")
# if driver_template is None:
#     raise FileNotFoundError("Cannot load Driver.png")
# h_out, w_out = driver_template.shape[:2]

# # Process each image in NS_Images_BEV
# for fname in os.listdir(bev_folder):
#     if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
#         continue

#     bev_path = os.path.join(bev_folder, fname)
#     bev_img  = cv2.imread(bev_path)
#     if bev_img is None:
#         print(f"⚠️ Skipping {fname}: failed to load")
#         continue

#     # Warp
#     warped = cv2.warpPerspective(bev_img, H_bev_to_driver, (w_out, h_out))

#     # Insert "_driver" before the extension
#     base, ext = os.path.splitext(fname)
#     out_fname = f"{base}_driver{ext}"
#     out_path = os.path.join(driver_folder, out_fname)

#     # Save
#     cv2.imwrite(out_path, warped)
#     print(f"✅ Saved {out_fname}")

# print("All done!")
