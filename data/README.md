# Dataset — AI4Mars

## About

AI4Mars is a dataset for terrain-aware autonomous driving on Mars, containing ~326K semantic segmentation labels on 35K images from the Curiosity, Opportunity, and Spirit rovers. Labels were collected through crowdsourcing with 10 annotators per image.

**Citation:**
> Swan, R. M., Atha, D., Leopold, H. A., et al. (2021). AI4Mars: A dataset for terrain-aware autonomous driving on Mars. In IEEE/CVF CVPR Workshops, pp. 1982–1991.

## Terrain Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | soil | Fine-grained regolith |
| 1 | bedrock | Exposed rock surface |
| 2 | sand | Sandy terrain, slip risk |
| 3 | big_rocks | Large rocks, obstacle risk |
| 255 | unknown | Unannotated regions |

## Pixel Distribution

| Class | Pixels | Percentage |
|-------|--------|------------|
| bedrock | 4,660,855,168 | 49.8% |
| soil | 3,485,135,841 | 37.2% |
| sand | 1,126,246,813 | 12.0% |
| big_rocks | 86,225,600 | 0.9% |

## Download Instructions

### Option 1 — Kaggle (recommended)
```bash
pip install kaggle
kaggle datasets download -d yash92328/ai4mars-terrainaware-autonomous-driving-on-mars --unzip -p data/ai4mars/
```

Requires Kaggle API credentials. See https://www.kaggle.com/settings → API → Create New Token.

### Option 2 — NASA Open Data Portal
https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars

## Expected Folder Structure After Download

```
data/ai4mars/
└── ai4mars-dataset-merged-0.1/
    └── msl/
        ├── images/
        │   └── edr/          ← 18,130 rover images (.JPG)
        └── labels/
            └── train/        ← 16,064 label masks (.png)
```

## Preprocessing

After downloading, build the good_labels.json file (filters out all-unknown labels):

```python
import os, json
import numpy as np
from PIL import Image

LABEL_DIR = 'data/ai4mars/ai4mars-dataset-merged-0.1/msl/labels/train'
files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.png')]
good  = [f for f in files if (np.array(Image.open(f'{LABEL_DIR}/{f}')) != 255).any()]
json.dump(good, open('data/good_labels.json', 'w'))
print(f'Saved {len(good)} good labels')  # Expected: ~15,901
```
