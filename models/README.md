# Model Weights

Pre-trained model weights are not committed to this repository due to GitHub file size limits.

## Available Weights

| File | Size | Description |
|------|------|-------------|
| `best_model.pt` | ~3.7 MB | MobileNetV3-Small trained on AI4Mars — 93.27% val accuracy |
| `dqn_rover.pt` | ~75 KB | Double DQN trained for 5,000 episodes — 94% mission completion (alpha=1.0, H_crit=0.8) |
| `class_index.json` | ~622 KB | Class index mapping for inference server |

## Retrain from scratch (recommended)

All models can be reproduced exactly using the provided scripts.

**Step 1 — Train MobileNetV3 terrain classifier (~30 min on cluster GPU):**
```bash
export CUDA_VISIBLE_DEVICES=4
python src/train.py \
    --img_dir data/ai4mars/ai4mars-dataset-merged-0.1/msl/images/edr \
    --label_dir data/ai4mars/ai4mars-dataset-merged-0.1/msl/labels/train \
    --good_json data/good_labels.json
```
Saves `models/best_model.pt` and `models/training_history.json`.

**Step 2 — Train RL agent (~40 min on cluster GPU):**
```bash
python src/train_rl.py \
    --save_dir models/ \
    --episodes 5000 \
    --grid_size 15
```
Saves `models/dqn_rover.pt` and `models/dqn_rover_final.pt`.

**Step 3 — Run ablations (~20 min):**
```bash
python src/run_ablations.py
```
Saves `ablation_results.json` to repo root.

## Usage

```python
import torch
from src.perception import TerrainClassifier

# Load perception model
tc = TerrainClassifier()
model = tc.get_model()
model.load_state_dict(torch.load('models/best_model.pt', map_location='cpu', weights_only=True))
model.eval()

# Load RL agent
from src.rl_agent import DoubleDQN
agent = DoubleDQN()
agent.online_net.load_state_dict(torch.load('models/dqn_rover.pt', map_location='cpu', weights_only=True))
agent.epsilon = 0.05
```
