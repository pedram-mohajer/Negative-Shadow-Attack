import cv2
import numpy as np
import torch
import math
import random
import os
import pandas as pd
from model import TwinLite as net

# === CONFIGURATION ===
METER_PER_PIXEL = 2.1 / 65
L_CHANNEL_BRIGHTNESS = 20
MIN_HEIGHT = 5
MAX_HEIGHT = 40
BOTTOM_MARGIN = 20
THRESHOLD = 0.10
NUM_TESTS = 100

# === HOMOGRAPHY: BEV → Driver View ===
pts_bev = np.float32([[6, 486], [6, 266], [1276, 270], [1276, 488]])
pts_driver = np.float32([[1120, 704], [7, 574], [588, 392], [674, 394]])
H_bev_to_driver, _ = cv2.findHomography(pts_bev, pts_driver)

P1, P2 = (50, 400), (1266, 400)
P3 = (50, 495)
image_path = "BEV.png"
BEV_IMG = cv2.imread(image_path)
if BEV_IMG is None:
    raise FileNotFoundError("BEV.png not found")

# === OUTPUT PREPARATION ===
os.makedirs("Detected", exist_ok=True)
os.makedirs("Not_Detected", exist_ok=True)

# === MODEL LOADING ===
model = net.TwinLiteNet()
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('pretrained/best.pth'))
model.eval()

def is_convex_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    cross = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    signs = [np.sign(cross(pts[i], pts[(i+1)%4], pts[(i+2)%4])) for i in range(4)]
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)

def generate_bright_patch(img, idx):
    for _ in range(100):
        rand_x = random.randint(P1[0], P2[0])
        y1 = random.randint(P1[1], P3[1] - BOTTOM_MARGIN)
        max_y2 = min(P3[1], y1 + MAX_HEIGHT)
        if max_y2 - y1 < MIN_HEIGHT:
            continue
        y2 = random.randint(y1 + MIN_HEIGHT, max_y2)
        delta_y = y2 - y1
        upper_limit = P3[1] - delta_y
        if y2 > upper_limit or upper_limit < y2:
            continue
        y3 = random.randint(y2, upper_limit)
        y4 = y3 + delta_y
        pt1, pt2 = (P1[0], y1), (P1[0], y2)
        pt3, pt4 = (rand_x, y4), (rand_x, y3)
        polygon = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
        if is_convex_quad(polygon):
            break
    else:
        return None

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    ys, xs = np.where(mask == 255)
    bright_pixels = list(zip(xs, ys))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = np.where(mask == 255, cv2.add(L, L_CHANNEL_BRIGHTNESS), L)
    lab = cv2.merge((np.clip(L, 0, 255).astype(np.uint8), A, B))
    bright_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    pt2, pt3 = (P1[0], y2), (rand_x, y3)
    length_m = round(math.hypot(pt3[0]-pt2[0], pt3[1]-pt2[1]) * METER_PER_PIXEL, 2)
    width_cm = round(delta_y * METER_PER_PIXEL * 100, 2)
    distance_cm = round((y1 - P1[1]) * METER_PER_PIXEL * 100, 2)
    angle_deg = round(math.degrees(math.atan2(pt3[1]-pt2[1], pt3[0]-pt2[0])), 2)

    return bright_img, bright_pixels, length_m, width_cm, distance_cm, angle_deg

def warp_pixels(pixels):
    if not pixels:
        return []
    pts = np.array(pixels, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H_bev_to_driver).reshape(-1, 2)
    return [tuple(map(int, pt)) for pt in transformed]

def group_lane_pixels(LL_mask):
    num_labels, labels = cv2.connectedComponents(LL_mask)
    groups = []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        coords = list(zip(xs.tolist(), ys.tolist()))
        groups.append(coords)
    return groups

def detect_lanes(model, img):
    img_resized = cv2.resize(img, (640, 360))
    tensor = img_resized[:, :, ::-1].transpose(2, 0, 1).copy()
    tensor = torch.from_numpy(tensor).unsqueeze(0).float().cuda() / 255.0
    with torch.no_grad():
        _, x1 = model(tensor)
    _, ll_predict = torch.max(x1, 1)
    LL = ll_predict.byte().cpu().numpy()[0] * 255
    return group_lane_pixels(LL), LL

# === MAIN LOOP ===
all_records = []
for i in range(NUM_TESTS):
    patch_result = generate_bright_patch(BEV_IMG.copy(), i)
    if patch_result is None:
        continue

    bright_img, bright_pixels, length, width, dist, angle = patch_result
    warped_img = cv2.warpPerspective(bright_img, H_bev_to_driver, (1280, 720))
    transformed_pixels = warp_pixels(bright_pixels)

    lane_groups, LL_mask = detect_lanes(model, warped_img)
    predicted_pixels = set(pt for group in lane_groups for pt in group)
    bright_set = set((int(x * 0.5), int(y * 0.5)) for x, y in transformed_pixels)
    intersection = predicted_pixels & bright_set
    ratio = len(intersection) / len(bright_set) if bright_set else 0

    result_img = warped_img.copy()
    result_img[LL_mask > 100] = [0, 255, 0]
    filename = f"sun_cast_{i}.png"
    save_dir = "Detected" if ratio > THRESHOLD else "Not_Detected"
    cv2.imwrite(os.path.join(save_dir, filename), result_img)

    all_records.append({
        "Image": filename,
        "Length(m)": length,
        "Width(cm)": width,
        "Distance(cm)": dist,
        "Angle(deg)": angle,
        "Detected": ratio > THRESHOLD,
        "Overlap(%)": round(ratio * 100, 2)
    })

# Save results
pd.DataFrame(all_records).to_csv("final_detection_results.csv", index=False)
print("✅ Pipeline finished and results saved.")
