# main.py — Search-Based Integration
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
from tqdm import tqdm
import pandas as pd

from cast_negative_shadow   import generate_shadow_polygon

from optimizer import run_search_optimization
from bev_to_driver import transform_bev_to_driver_view
from twinlitenet import run_twinlitenet
from hybridnet import run_hybridnets
from clrernet import run_clrernet

from mmdet.apis import init_detector

CSV_PATH = "NS_Images_BEV/shadow_lengths.csv"

# ────────────────────────────────────────────────────────────
def load_bev_image(image_path: str = "./BEV.png") -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"BEV template not found: {image_path}")
    print("✅ BEV image path verified.")
    return image_path

# ────────────────────────────────────────────────────────────
def main(n_samples: int = 10000, search_based: bool = False) -> None: # 100000

    model = init_detector(
        "./clrer/configs/clrernet/culane/clrernet_culane_dla34_ema.py",
        "./clrer/configs/clrernet_culane_dla34_ema.pth",
        device='cuda:0'
    )

    image_path = load_bev_image()

    twinlitenet_results: list[bool] = []
    hybridnet_results: list[bool] = []
    clrernet_results: list[bool] = []

    if search_based:
        print("🚀 Running search-based optimization...")
        run_search_optimization(n_samples, model)
    else:
        for i in tqdm(range(n_samples), desc="Processing shadow IDs"):
            px_bev = generate_shadow_polygon(image_path, shadow_id=i)
            px_drv = transform_bev_to_driver_view(px_bev, shadow_id=i)

            twinlitenet_detected, _ = run_twinlitenet(px_drv, shadow_id=i)
            hybridnet_detected, _ = run_hybridnets(px_drv, shadow_id=i)
            clrernet_detected, _   = run_clrernet(model, px_drv, shadow_id=i)


            twinlitenet_results.append(twinlitenet_detected)
            hybridnet_results.append(hybridnet_detected)
            clrernet_results.append(clrernet_detected)

        df = pd.read_csv(CSV_PATH)

        if len(df) != len(twinlitenet_results):
            raise ValueError("CSV row count does not match number of samples.")

        df["twinlitenets"] = twinlitenet_results
        df["hybridnets"] = hybridnet_results
        df["clrernet"] = clrernet_results

        df.to_csv(CSV_PATH, index=False)
        print("📄 CSV updated with Detected column.")



    os.system(f"cp {CSV_PATH} results/result.csv")
    print("📄 CSV copied to results/result.csv")

    import torch
    del model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(search_based=True)
