# bev_to_driver.py
# ----------------
# Convert a BEV shadow image and its bright-pixel list to driver view,
# save the warped PNG, return the transformed pixel list, and log only
# the scalar metadata to CSV (no huge pixel arrays written to disk).

import os
import cv2
import numpy as np
import pandas as pd

# ────────── paths & folders ──────────
bev_folder    = "NS_Images_BEV"
driver_folder = "NS_Images_Driver"
os.makedirs(driver_folder, exist_ok=True)

# ────────── homography (fixed) ───────
pts_bev = np.float32([[6, 486], [6, 266], [1276, 270], [1276, 488]])
pts_drv = np.float32([[1120, 704], [7, 574], [588, 392], [674, 394]])
H_bev2drv, _ = cv2.findHomography(pts_bev, pts_drv)

# output image size → from template
template = cv2.imread("Driver.png")
if template is None:
    raise FileNotFoundError("Driver.png not found.")
h_out, w_out = template.shape[:2]

from typing import List, Tuple

# ─────────────────────────────────────
def transform_bev_to_driver_view(pixel_list_bev: List[Tuple[int, int]],
                                 shadow_id: int) -> List[Tuple[int, int]]:

    """
    Parameters
    ----------
    pixel_list_bev : list[(x,y)]
        Bright-area pixels in BEV coordinates (as produced by
        generate_shadow_polygon()).
    shadow_id : int
        Sequential id, shared with the BEV file name.

    Returns
    -------
    pixel_list_driver : list[(x,y)]
        Same pixels warped into the driver-view frame.
    """

    # 1) load BEV image -----------------------------------------------------
    fname_bev = f"sun_cast_{shadow_id}.png"
    bev_path  = os.path.join(bev_folder, fname_bev)
    bev_img   = cv2.imread(bev_path)
    if bev_img is None:
        raise FileNotFoundError(f"BEV image not found: {bev_path}")

    # 2) warp full image ----------------------------------------------------
    drv_img = cv2.warpPerspective(bev_img, H_bev2drv, (w_out, h_out))
    fname_drv = f"sun_cast_{shadow_id}.png"
    drv_path  = os.path.join(driver_folder, fname_drv)
    cv2.imwrite(drv_path, drv_img)
    #print(f"✅ Saved warped image: {fname_drv}")

    # 3) warp bright-pixel list --------------------------------------------
    if pixel_list_bev:
        pts = np.asarray(pixel_list_bev, np.float32).reshape(-1, 1, 2)
        pts_drv = cv2.perspectiveTransform(pts, H_bev2drv).reshape(-1, 2)
        pixel_list_drv = [tuple(map(int, p)) for p in pts_drv]
    else:
        pixel_list_drv = []

    # 4) log scalar metadata (no big lists) --------------------------------
    csv_bev = os.path.join(bev_folder, "shadow_lengths.csv")
    if not os.path.exists(csv_bev):
        raise FileNotFoundError(csv_bev)

    # pick the row for this shadow_id and copy only scalar cols
    df_bev = pd.read_csv(csv_bev)
    row    = df_bev.loc[df_bev["Image_Name"] == fname_bev]
    if row.empty:
        raise ValueError(f"No CSV entry for {fname_bev}")

    cols_keep = ["Image_Name", "Length(m)", "Width(cm)",
                 "Distance(cm)", "Angle(deg)"]
    record = row.iloc[0][cols_keep].to_dict()
    record["Image_Name"] = fname_drv      # overwrite with driver name

    csv_drv = os.path.join(driver_folder, "shadow_lengths_driver.csv")
    if os.path.exists(csv_drv):
        df_drv = pd.read_csv(csv_drv)
        df_drv.loc[len(df_drv)] = record
    else:
        df_drv = pd.DataFrame([record])
    df_drv.to_csv(csv_drv, index=False)
    #print("📄 CSV (scalar) updated.")

    # 5) hand the transformed list back to the caller ----------------------
    return pixel_list_drv

