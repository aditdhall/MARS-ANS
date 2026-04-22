"""Download the AI4Mars dataset and build good_labels.json.

Usage:
    python src/setup_dataset.py
    python src/setup_dataset.py --skip-download   # only rebuild good_labels.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AI4MARS_DIR = DATA_DIR / "ai4mars"
LABEL_DIR = AI4MARS_DIR / "ai4mars-dataset-merged-0.1" / "msl" / "labels" / "train"
IMAGE_DIR = AI4MARS_DIR / "ai4mars-dataset-merged-0.1" / "msl" / "images" / "edr"
GOOD_LABELS_JSON = DATA_DIR / "good_labels.json"

KAGGLE_SLUG = "yash92328/ai4mars-terrainaware-autonomous-driving-on-mars"


def download_kaggle():
    AI4MARS_DIR.mkdir(parents=True, exist_ok=True)
    if LABEL_DIR.exists() and any(LABEL_DIR.iterdir()):
        print(f"Dataset already present at {AI4MARS_DIR}, skipping download.")
        return
    print(f"Downloading AI4Mars from Kaggle to {AI4MARS_DIR} ...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_SLUG,
        "--unzip",
        "-p", str(AI4MARS_DIR),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit(
            "kaggle CLI not found. Install with `pip install kaggle` and configure "
            "credentials at ~/.kaggle/kaggle.json (https://www.kaggle.com/settings)."
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"Kaggle download failed (exit {e.returncode}). See message above.")


def build_good_labels():
    if not LABEL_DIR.exists():
        sys.exit(f"Label directory not found: {LABEL_DIR}")
    files = sorted(f for f in os.listdir(LABEL_DIR) if f.endswith(".png"))
    print(f"Scanning {len(files)} label masks ...")
    good = []
    for i, f in enumerate(files, 1):
        arr = np.array(Image.open(LABEL_DIR / f))
        if (arr != 255).any():
            good.append(f)
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}  kept {len(good)}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(GOOD_LABELS_JSON, "w") as fh:
        json.dump(good, fh)
    print(f"Saved {len(good)} good labels to {GOOD_LABELS_JSON}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip Kaggle download and only rebuild good_labels.json.")
    args = ap.parse_args()

    if not args.skip_download:
        download_kaggle()
    build_good_labels()
    print("Done.")
    print(f"  images: {IMAGE_DIR}")
    print(f"  labels: {LABEL_DIR}")
    print(f"  index:  {GOOD_LABELS_JSON}")


if __name__ == "__main__":
    main()
