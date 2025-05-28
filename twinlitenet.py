"""
twinlitenet.py – lane-vs-bright-area overlap checker
====================================================

• Accepts bright pixels *directly* as an argument (driver space, 1280 × 720)
• Resizes driver frame to 640 × 360 for TwinLite and scales the bright pixels
  the same way.
• Writes colourised result to results/Detected/ or results/Undetected/.
"""

from __future__ import annotations
from typing import List, Sequence, Tuple, Set

import os
import cv2
import numpy as np
import torch
from twinlite.model import TwinLite as net   # ← your network package


# ────────────── lazy-loaded model ───────────────────────────
_model: torch.nn.Module | None = None


# ────────────── helper: connected components ───────────────
def _group_lane_pixels(mask: np.ndarray) -> List[Sequence[Tuple[int, int]]]:
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    return [list(zip(*np.where(labels == lab)[::-1]))
            for lab in range(1, num_labels)]


# ────────────── helper: forward through TwinLite ───────────
def _run_twinlite(model: torch.nn.Module,
                  frame_bgr: np.ndarray
                  ) -> Tuple[np.ndarray,
                             List[Sequence[Tuple[int, int]]]]:
    """Resize to 640 × 360, run model, overlay mask."""
    img640 = cv2.resize(frame_bgr, (640, 360))
    tensor = (torch.from_numpy(img640[:, :, ::-1]      # BGR→RGB
                               .transpose(2, 0, 1).copy())
              .unsqueeze(0).cuda().float() / 255.0)

    with torch.no_grad():
        _, logits = model(tensor)
    _, pred = torch.max(logits, 1)
    mask = (pred.byte().cpu().numpy()[0] * 255).astype(np.uint8)

    vis = img640.copy()
    vis[mask == 255] = (0, 255, 0)                     # green lane
    return vis, _group_lane_pixels(mask)


# ────────────── public API ─────────────────────────────────
def run_twinlitenet(bright_pixels_driver: list[tuple[int, int]], shadow_id: int) -> Tuple[bool, float]:

    """
    Parameters
    ----------
    bright_pixels_driver : list[(x, y)]
        Bright-area pixels already warped into native driver view
        (1280 × 720).
    shadow_id : int
        Same id used in the file name  sun_cast_<id>.png .
    """

    global _model

    # 1) ensure output dirs
    os.makedirs("results/TwinLiteNet/Detected",   exist_ok=True)
    os.makedirs("results/TwinLiteNet/Undetected", exist_ok=True)

    # 2) lazy-load TwinLite
    if _model is None:
        #print("📦 Loading TwinLite …")
        mdl = net.TwinLiteNet()
        mdl = torch.nn.DataParallel(mdl).cuda()
        mdl.load_state_dict(torch.load("twinlite/pretrained/best.pth"))
        mdl.eval()
        _model = mdl

    # 3) load driver-view PNG
    img_name = f"sun_cast_{shadow_id}.png"
    img_path = os.path.join("NS_Images_Driver", img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        #print(f"❌ Could not load {img_path}")
        return
    h0, w0 = frame.shape[:2]          # should be 720 × 1280

    # 4) network inference
    vis, lane_blobs = _run_twinlite(_model, frame)
    lane_px: Set[Tuple[int, int]] = {pt for g in lane_blobs for pt in g}

    # 5) scale bright pixels to 640 × 360
    sx, sy = 640 / w0, 360 / h0
    bright_px: Set[Tuple[int, int]] = {
        (int(round(x * sx)), int(round(y * sy)))
        for x, y in bright_pixels_driver
    }

    # 6) overlap calculation
    inter   = lane_px & bright_px
    overlap = len(inter) / len(bright_px) if bright_px else 0

    #print(f"\n🖼 {img_name} | bright:{len(bright_px):5d}  " f"mask:{len(lane_px):5d}  → inter:{len(inter):5d} " f"({overlap:.2%})")

    category = "Detected" if overlap > 0.10 else "Undetected"
    #print("✅ Detected" if category == "Detected" else "❌ Undetected")

    # 7) save colourised result
    out_path = os.path.join("results/TwinLiteNet", category, img_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)
    #print(f"📁 Saved: {out_path}")
    return category == "Detected", overlap
    

