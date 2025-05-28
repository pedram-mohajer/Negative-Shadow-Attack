# 🕳️ Negative Shadow Attack: Adversarial Bright Patterns for Lane Detection Deception

This repository implements the **Negative Shadow (NS) attack**—a stealthy, non-invasive adversarial method that targets **Automated Lane Centering (ALC)** systems in Level 2 autonomous vehicles. By casting sunlight through physical occluders (e.g., canopies or fences), the attack produces bright, lane-like patterns that exploit intensity bias in LD preprocessing. These patterns cause the ALC system to misinterpret them as valid lane markings, leading to off-path deviations or unsafe maneuvers—without modifying the road or vehicle.

---

## 🔧 Installation (Docker Recommended)

To ensure reproducibility and GPU support, we strongly recommend running in a Docker environment.

### 1. Install Docker & Compose

```bash
sudo apt install docker-compose
sudo service docker start
```

### 2. Enable NVIDIA Runtime

Update `/etc/docker/daemon.json`:

```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

Then restart Docker:

```bash
sudo systemctl restart docker
```

### 3. Build and Run the Container

```bash
docker compose build --build-arg UID="$(id -u)" dev
docker compose run --rm dev
```

---

## 🚀 Project Overview

This pipeline simulates and evaluates the Negative Shadow attack under two modes:

- **Random Mode**: generate `n` random shadows and test three LD models.
- **Search Mode**: apply a Genetic Algorithm to find adversarial NS parameters.

### Supported Lane Detectors:
- **TwinLiteNet** (lightweight, feature-based)
- **HybridNets** (multi-task perception network)
- **CLRerNet** (confidence-regularized)

---

## ▶️ Running the Pipeline

### Default Entry: `main.py`

```bash
python main.py
```

By default, `main.py` runs search-based optimization. To run random sampling instead:

```python
if __name__ == "__main__":
    main(n_samples=1000, search_based=False)
```

### What It Does
- Generates NS images in BEV (`NS_Images_BEV/`)
- Warps them to driver-view (`NS_Images_Driver/`)
- Evaluates lane overlap using all three detectors
- Stores metadata in `shadow_lengths.csv`
- Outputs visual results and detection flags to `results/`

---

## 📁 Directory Structure

```
NS_Images_BEV/         # Bright regions in top-down BEV
NS_Images_Driver/      # Warped driver-view images
results/
├── TwinLiteNet/
├── HybridNets/
├── CLRerNet/
└── result.csv          # Final CSV summary with detection flags

pretrained/
├── twinlite/best.pth
├── hybrid/weights/
└── clrer/configs/clrernet_culane_dla34_ema.pth
```

---

## 🧬 Genetic Optimization (Search Mode)

`optimizer.py` runs a Genetic Algorithm to evolve high-impact shadow parameters.

Fitness is based on:
- **Detector Fooling**: overlap between NS region and predicted lanes
- **Parameter Tuning**: Encourages shadows with width \(W\), length \(L\), distance \(D\), and angle \(\beta\) values that align with ranges empirically shown to confuse LD models.

Top candidates are retained and mutated over generations to find stealthy yet effective shadows.


## 🛡️ Disclaimer

This research is intended solely for academic and ethical security analysis. All experiments were run in simulation or controlled conditions. The NS attack demonstrates vulnerabilities in lane detection pipelines and emphasizes the need for robust defenses in AV perception systems.

---

##  📝 Acknowledgments

This work evaluates the robustness of several lane detection models as part of our adversarial attack study. We gratefully acknowledge the authors and open-source implementations of the following models:

```
@article{che2023twinlitenet,
  title={TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars},
  author={Che, Quang Huy and Nguyen, Dinh Phuc and Pham, Minh Quan and Lam, Duc Khai},
  journal={arXiv preprint arXiv:2307.10705},
  year={2023}
}
@article{honda2023clrernet,
  title={CLRerNet: Improving Confidence of Lane Detection with LaneIoU},
  author={Honda, Hiroto and Uchida, Yusuke},
  journal={arXiv preprint arXiv:2305.08366},
  year={2023}
}
@article{vu2022hybridnets,
  title={Hybridnets: End-to-end perception network},
  author={Vu, Dat and Ngo, Bao and Phan, Hung},
  journal={arXiv preprint arXiv:2203.09035},
  year={2022}
}
```
