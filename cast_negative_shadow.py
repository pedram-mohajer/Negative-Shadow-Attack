# cast_negative_shadow.py — Search-Based Version
import cv2
import numpy as np
import math
import os
import pandas as pd
import random


# Adjustable Constants
L_CHANNEL_BRIGHTNESS = 20
METER_PER_PIXEL = 2.1 / 65
OUTPUT_DIR = "NS_Images_BEV"
CSV_PATH = os.path.join(OUTPUT_DIR, "shadow_lengths.csv")

P1, P2 = (50, 400), (1266, 400)
P3, P4 = (50, 495), (1266, 495)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_convex_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    cross = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    signs = [np.sign(cross(pts[i], pts[(i+1)%4], pts[(i+2)%4])) for i in range(4)]
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)

def pixel_distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def create_shadow_image(original_img, polygon, shadow_id):
    img = original_img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    ys, xs = np.where(mask == 255)
    pixel_list = list(zip(xs.tolist(), ys.tolist()))

    # LAB brightening
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_img)
    L = np.where(mask == 255, cv2.add(L, L_CHANNEL_BRIGHTNESS), L)
    L = np.clip(L, 0, 255).astype(np.uint8)
    lab_bright = cv2.merge((L, A, B))
    img_bright = cv2.cvtColor(lab_bright, cv2.COLOR_LAB2BGR)

    # Save
    image_name = f"sun_cast_{shadow_id}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, image_name), img_bright.astype(np.uint8))

    return image_name, pixel_list

def record_metadata(image_name, pt2, pt3, delta_y, y1):
    length_m = round(pixel_distance(pt2, pt3) * METER_PER_PIXEL, 2)
    width_cm = round(delta_y * METER_PER_PIXEL * 100, 2)
    distance_cm = round((y1 - P1[1]) * METER_PER_PIXEL * 100, 2)
    #angle_deg = round(math.degrees(math.atan2(pt3[1]-pt2[1], pt3[0]-pt2[0])), 2)
    angle_deg = abs(round(math.degrees(math.atan2(pt3[1] - pt2[1], pt3[0] - pt2[0])), 2))


    return {
        "Image_Name": image_name,
        "Length(m)": length_m,
        "Width(cm)": width_cm,
        "Distance(cm)": distance_cm,
        "Angle(deg)": angle_deg
    }

def save_to_csv(record):
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(CSV_PATH, index=False)


def generate_shadow(image_path, shadow_id, y1=None, y2=None, y3=None, rand_x=None):
    original_img = cv2.imread(image_path)
    if original_img is None:
        return []

    MIN_HEIGHT = 5
    MAX_HEIGHT = 40
    BOTTOM_MARGIN = 20
    min_pixel_width = int(np.ceil(4 / 100 / METER_PER_PIXEL))
    max_attempts = 1000

    for _ in range(max_attempts):
        # Resample everything fresh each time if not fixed
        this_y1     = y1 if y1 is not None else random.randint(P1[1], P3[1] - BOTTOM_MARGIN)
        height      = random.randint(MIN_HEIGHT, MAX_HEIGHT)
        this_y2     = y2 if y2 is not None else this_y1 + height
        delta_y     = this_y2 - this_y1
        rand_x_curr = rand_x if rand_x is not None else random.randint(P1[0], P2[0])

        if delta_y < min_pixel_width:
            continue

        upper_limit = P3[1] - delta_y
        if this_y2 > upper_limit:
            continue

        try:
            this_y3 = y3 if y3 is not None else random.randint(this_y2, upper_limit)
        except ValueError:
            continue

        y4 = this_y3 + delta_y
        pt1, pt2 = (P1[0], this_y1), (P1[0], this_y2)
        pt3, pt4 = (rand_x_curr, y4), (rand_x_curr, this_y3)
        polygon = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)

        if not is_convex_quad(polygon):
            continue

        image_name, px_list = create_shadow_image(original_img, polygon, shadow_id)
        record = record_metadata(image_name, pt2, pt3, delta_y, this_y1)
        save_to_csv(record)
        return px_list

    print(f"⚠️ Failed to inject shadow after {max_attempts} attempts for shadow_id={shadow_id}")
    return []




def generate_shadow_polygon(image_path, shadow_id):
    # ===================== ADJUSTABLE PARAMETERS =====================
    L_CHANNEL_BRIGHTNESS = 20
    MIN_HEIGHT = 5
    MAX_HEIGHT = 40
    BOTTOM_MARGIN = 20
    METER_PER_PIXEL = 2.1 / 65
    OUTPUT_DIR = "NS_Images_BEV"
    CSV_PATH = os.path.join(OUTPUT_DIR, "shadow_lengths.csv")
    # ================================================================

    P1, P2 = (50, 400), (1266, 400)
    P3, P4 = (50, 495), (1266, 495)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    def is_convex_quad(pts):
        pts = np.array(pts, dtype=np.float32)
        cross = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        signs = [np.sign(cross(pts[i], pts[(i+1)%4], pts[(i+2)%4])) for i in range(4)]
        return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)

    def pixel_distance(p1, p2):
        return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

    # Start generating until a convex polygon is found
    while True:
        img = original_img.copy()
        rand_x = random.randint(P1[0], P2[0])

        for _ in range(100):  # Try max 100 times per generation
            y1 = random.randint(P1[1], P3[1] - BOTTOM_MARGIN)
            y2 = random.randint(y1 + MIN_HEIGHT, min(P3[1], y1 + MAX_HEIGHT))
            
            # delta_y = y2 - y1
            # upper_limit = P3[1] - delta_y
            # if y2 > upper_limit:
            #     continue
            ##################################
            delta_y = y2 - y1
            min_pixel_width = int(np.ceil(4 / 100 / METER_PER_PIXEL))  # ≈ 2 pixels
            if delta_y < min_pixel_width:
                continue  # skip if width too small

            upper_limit = P3[1] - delta_y
            if y2 > upper_limit:
                continue
            ##################################

            y3 = random.randint(y2, upper_limit)
            y4 = y3 + delta_y
            pt1, pt2 = (P1[0], y1), (P1[0], y2)
            pt3, pt4 = (rand_x, y4), (rand_x, y3)
            polygon = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
            if is_convex_quad(polygon):
                break
        else:
            continue  # Retry whole generation if no convex polygon found

        # ===== Create mask and get pixel list =====
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        ys, xs = np.where(mask == 255)
        pixel_list = list(zip(xs.tolist(), ys.tolist()))

        # ===== LAB brightening =====
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab_img)
        L = np.where(mask == 255, cv2.add(L, L_CHANNEL_BRIGHTNESS), L)
        L = np.clip(L, 0, 255).astype(np.uint8)
        lab_bright = cv2.merge((L, A, B))
        img_bright = cv2.cvtColor(lab_bright, cv2.COLOR_LAB2BGR)

        # ===== Metric calculations =====
        pixel_len = pixel_distance(pt2, pt3)
        length_m = round(pixel_len * METER_PER_PIXEL, 2)
        width_cm = round(delta_y * METER_PER_PIXEL * 100, 2)
        vertical_pixel_diff = y1 - P1[1]
        distance_cm = round(vertical_pixel_diff * METER_PER_PIXEL * 100, 2)
        angle_deg = round(math.degrees(math.atan2(pt3[1]-pt2[1], pt3[0]-pt2[0])), 2)

        image_name = f"sun_cast_{shadow_id}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, image_name), img_bright.astype(np.uint8))

        # ===== Append to CSV =====
        record = {
            "Image_Name": image_name,
            "Length(m)": length_m,
            "Width(cm)": width_cm,
            "Distance(cm)": distance_cm,
            "Angle(deg)": angle_deg
            #"BrightPixels": pixel_list
        }

        # Append to existing or create new CSV
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            df = pd.DataFrame([record])
        df.to_csv(CSV_PATH, index=False)
        break
    return pixel_list


