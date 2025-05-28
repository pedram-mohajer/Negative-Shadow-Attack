import cv2
import numpy as np
import random
import math
import os
import pandas as pd
from scipy.ndimage import distance_transform_edt

# ===================== ADJUSTABLE PARAMETERS =====================
L_CHANNEL_BRIGHTNESS = 20
GRADIENT_GLOW_STRENGTH = 23
OVERLAY_BLEND_STRENGTH = 0.15
MIN_HEIGHT = 5
MAX_HEIGHT = 40
BOTTOM_MARGIN = 20
# ================================================================

image_path = "./BEV.png"
original_img = cv2.imread(image_path)
if original_img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

P1 = (50, 400)
P3 = (50, 495)
P2 = (1266, 400)
P4 = (1266, 495)

METER_PER_PIXEL = 2.1 / 65
os.makedirs("NS_Images_BEV", exist_ok=True)
record_list = []
shadow_id = 0

def is_convex_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    cross = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    signs = [np.sign(cross(pts[i], pts[(i+1)%4], pts[(i+2)%4])) for i in range(4)]
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)

def pixel_distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def shift_channel(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

while True:
    img = original_img.copy()
    rand_x = random.randint(P1[0], P2[0])

    # generate polygon
    for _ in range(100):
        y1 = random.randint(P1[1], P3[1] - BOTTOM_MARGIN)
        y2 = random.randint(y1 + MIN_HEIGHT, min(P3[1], y1 + MAX_HEIGHT))
        delta_y = y2 - y1
        upper_limit = P3[1] - delta_y
        if y2 > upper_limit:
            continue
        y3 = random.randint(y2, upper_limit)
        y4 = y3 + delta_y
        pt1, pt2 = (P1[0], y1), (P1[0], y2)
        pt3, pt4 = (rand_x, y4), (rand_x, y3)
        polygon = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
        if is_convex_quad(polygon):
            break
    else:
        print("❌ Failed to generate convex polygon.")
        continue

    # ===== build pixel‐mask and extract coords =====
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)

    # this is the new bit: collect all (x, y) inside the polygon
    ys, xs = np.where(mask == 255)
    pixel_list = list(zip(xs.tolist(), ys.tolist()))

    # ===== LAB Brightening =====
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_img)
    L = np.where(mask == 255, cv2.add(L, L_CHANNEL_BRIGHTNESS), L)
    L = np.clip(L, 0, 255).astype(np.uint8)
    lab_bright = cv2.merge((L, A, B))
    img_bright = cv2.cvtColor(lab_bright, cv2.COLOR_LAB2BGR)

    # … your existing sunlight, gradient, texture, blur, etc. …

    # At the end, compute your metrics:
    pixel_len = pixel_distance(pt2, pt3)
    length_m = round(pixel_len * METER_PER_PIXEL, 2)
    width_cm = round(delta_y * METER_PER_PIXEL * 100, 2)
    vertical_pixel_diff = y1 - P1[1]
    distance_cm = round(vertical_pixel_diff * METER_PER_PIXEL * 100, 2)
    angle_deg = round(math.degrees(math.atan2(pt3[1]-pt2[1], pt3[0]-pt2[0])), 2)

    image_name = f"sun_cast_{shadow_id}.png"
    cv2.imwrite(os.path.join("NS_Images_BEV", image_name), img_bright.astype(np.uint8))

    # print(pixel_list)
    # exit(0)

    # ===== append to your record list, including the pixel_list =====
    record_list.append({
        "Image_Name": image_name,
        "Length(m)": length_m,
        "Width(cm)": width_cm,
        "Distance(cm)": distance_cm,
        "Angle(deg)": angle_deg,
        "BrightPixels": pixel_list          # ← this column will hold your list
    })

    # ===== save CSV after each iteration =====
    pd.DataFrame(record_list).to_csv(
        "NS_Images_BEV/shadow_lengths.csv",
        index=False
    )

    cv2.imshow("Sunlight Cast on Street", img_bright.astype(np.uint8))
    key = cv2.waitKey(0)
    if key == 27:        # ESC to exit
        break
    elif key == 32:      # SPACE to next
        shadow_id += 1
        continue

cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import random
# import math
# import os
# import pandas as pd
# from scipy.ndimage import distance_transform_edt

# # ===================== ADJUSTABLE PARAMETERS =====================
# L_CHANNEL_BRIGHTNESS = 20
# GRADIENT_GLOW_STRENGTH = 23
# OVERLAY_BLEND_STRENGTH = 0.15
# MIN_HEIGHT = 5
# MAX_HEIGHT = 40
# BOTTOM_MARGIN = 20
# # ================================================================

# image_path = "./BEV.png"
# original_img = cv2.imread(image_path)
# if original_img is None:
#     raise FileNotFoundError(f"Image not found at path: {image_path}")

# P1 = (50, 400)
# P3 = (50, 495)
# P2 = (1266, 400)
# P4 = (1266, 495)

# METER_PER_PIXEL = 2.1 / 65
# os.makedirs("NS_Images", exist_ok=True)
# record_list = []
# shadow_id = 0

# def is_convex_quad(pts):
#     pts = np.array(pts, dtype=np.float32)
#     cross = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
#     signs = [np.sign(cross(pts[i], pts[(i+1)%4], pts[(i+2)%4])) for i in range(4)]
#     return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)

# def pixel_distance(p1, p2):
#     return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

# def shift_channel(img, dx, dy):
#     M = np.float32([[1, 0, dx], [0, 1, dy]])
#     return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# while True:
#     img = original_img.copy()
#     rand_x = random.randint(P1[0], P2[0])

#     for _ in range(100):
#         y1 = random.randint(P1[1], P3[1] - BOTTOM_MARGIN)
#         y2 = random.randint(y1 + MIN_HEIGHT, min(P3[1], y1 + MAX_HEIGHT))
#         delta_y = y2 - y1
#         upper_limit = P3[1] - delta_y
#         if y2 > upper_limit:
#             continue
#         y3 = random.randint(y2, upper_limit)
#         y4 = y3 + delta_y
#         pt1, pt2 = (P1[0], y1), (P1[0], y2)
#         pt3, pt4 = (rand_x, y4), (rand_x, y3)
#         polygon = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
#         if is_convex_quad(polygon):
#             break
#     else:
#         print("❌ Failed to generate convex polygon.")
#         continue

#     # ===== LAB Brightening =====
#     lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     L, A, B = cv2.split(lab_img)
#     mask = np.zeros(img.shape[:2], dtype=np.uint8)
#     cv2.fillConvexPoly(mask, polygon, 255)
#     L = np.where(mask == 255, cv2.add(L, L_CHANNEL_BRIGHTNESS), L)
#     L = np.clip(L, 0, 255).astype(np.uint8)
#     lab_bright = cv2.merge((L, A, B))
#     img_bright = cv2.cvtColor(lab_bright, cv2.COLOR_LAB2BGR)

#     # ===== Feathered Sunlight Overlay =====
#     blur_radius = 10
#     edge_mask = np.zeros(img.shape[:2], dtype=np.uint8)
#     cv2.fillConvexPoly(edge_mask, polygon, 255)
#     feathered_mask = cv2.GaussianBlur(edge_mask, (2 * blur_radius + 1, 2 * blur_radius + 1), 0)
#     sun_overlay = np.full_like(img, (180, 210, 255))
#     alpha_mask = feathered_mask.astype(np.float32) / 255.0
#     alpha_mask = np.stack([alpha_mask]*3, axis=2)

#     # ===== Perlin-like Opacity Noise =====
#     noise = np.random.normal(0.0, 0.15, size=mask.shape[:2])
#     noise = cv2.GaussianBlur(noise, (15, 15), 0)
#     alpha_mask *= (1 + noise[..., None])
#     alpha_mask = np.clip(alpha_mask, 0, 1)

#     img_bright = img_bright.astype(np.float32)
#     img_bright = (1 - alpha_mask * OVERLAY_BLEND_STRENGTH) * img_bright + (alpha_mask * OVERLAY_BLEND_STRENGTH) * sun_overlay
#     img_bright = np.clip(img_bright, 0, 255)

#     # ===== Directional Gradient Falloff =====
#     center_mask = np.zeros_like(mask, dtype=np.uint8)
#     cv2.fillConvexPoly(center_mask, polygon, 255)
#     h, w = mask.shape
#     Y, X = np.mgrid[0:h, 0:w]
#     sun_direction = np.array([1.0, 0.5])  # Simulated sunlight direction
#     gradient_dir = (X * sun_direction[0] + Y * sun_direction[1])
#     gradient_dir = (gradient_dir - np.min(gradient_dir)) / (np.max(gradient_dir) - np.min(gradient_dir))
#     gradient_dir = 1 - gradient_dir  # Flip for closer to light = brighter

#     dist_transform = distance_transform_edt(center_mask)
#     max_dist = np.max(dist_transform)
#     grad_mask = 1 - (dist_transform / max_dist)
#     grad_mask = np.clip(grad_mask, 0, 1)
#     final_grad = grad_mask * gradient_dir
#     grad_mask_weighted = np.stack([final_grad]*3, axis=2) * (center_mask[:, :, None] / 255.0)
#     img_bright += grad_mask_weighted * GRADIENT_GLOW_STRENGTH
#     img_bright = np.clip(img_bright, 0, 255)

#     # ===== Micro-texture Overlay (optional) =====
#     if os.path.exists("sun_noise.png"):
#         texture = cv2.imread("sun_noise.png").astype(np.float32) / 255.0
#         texture = cv2.resize(texture, (img.shape[1], img.shape[0]))
#         img_bright = img_bright * (1 - alpha_mask * 0.1) + texture * alpha_mask * 0.1
#         img_bright = np.clip(img_bright, 0, 255)

#     # ===== Chromatic Aberration =====
#     b, g, r = cv2.split(img_bright.astype(np.uint8))
#     b = shift_channel(b, -1, 0)
#     r = shift_channel(r, 1, 0)
#     img_bright = cv2.merge([b, g, r]).astype(np.float32)

#     # ===== Dynamic Directional Blur =====
#     dy = pt3[1] - pt2[1]
#     dx = pt3[0] - pt2[0]
#     angle_rad = math.atan2(dy, dx)
#     k_size = 9
#     kernel = np.zeros((k_size, k_size))
#     center = k_size // 2
#     dir_x = int(round(math.cos(angle_rad)))
#     dir_y = int(round(math.sin(angle_rad)))
#     for i in range(k_size):
#         x = center + dir_x * (i - center)
#         y = center + dir_y * (i - center)
#         if 0 <= x < k_size and 0 <= y < k_size:
#             kernel[y, x] = 1
#     kernel /= np.sum(kernel)
#     img_bright = cv2.filter2D(img_bright.astype(np.uint8), -1, kernel)

#     # ===== Metrics & Save =====
#     pixel_len = pixel_distance(pt2, pt3)
#     length_m = round(pixel_len * METER_PER_PIXEL, 2)
#     width_cm = round(delta_y * METER_PER_PIXEL * 100, 2)
#     vertical_pixel_diff = y1 - P1[1]
#     distance_cm = round(vertical_pixel_diff * METER_PER_PIXEL * 100, 2)
#     angle_deg = round(math.degrees(angle_rad), 2)

#     image_name = f"sun_cast_{shadow_id}.png"
#     cv2.imwrite(os.path.join("NS_Images_BEV", image_name), img_bright.astype(np.uint8))

#     record_list.append({
#         "Image_Name": image_name,
#         "Length(m)": length_m,
#         "Width(cm)": width_cm,
#         "Distance(cm)": distance_cm,
#         "Angle(deg)": angle_deg
#     })

#     pd.DataFrame(record_list).to_csv("NS_Images_BEV/shadow_lengths.csv", index=False)

#     cv2.imshow("Sunlight Cast on Street", img_bright.astype(np.uint8))
#     key = cv2.waitKey(0)
#     if key == 27:
#         break
#     elif key == 32:
#         shadow_id += 1
#         continue

# cv2.destroyAllWindows()


