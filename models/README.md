# Model Weights

Pre-trained model weights are stored in Google Drive due to GitHub file size limits.

## Available Weights

| File | Size | Description | Val Accuracy |
|------|------|-------------|-------------|
| `best_model.pt` | ~9.8 MB | MobileNetV3-Small fine-tuned on AI4Mars | 92.83% |
| `dqn_rover.pt` | ~0.3 MB | Double DQN trained for 500 episodes | 100% completion (alpha=0.75) |

## Download Instructions

### Option 1 — Google Drive (recommended)
Contact the team for Drive access link. Download both files and place them in this `models/` directory.

### Option 2 — Retrain from scratch
**MobileNetV3 (requires Colab A100, ~25 min):**
```bash
python src/train.py \
    --img_dir /path/to/ai4mars/msl/images/edr \
    --label_dir /path/to/ai4mars/msl/labels/train \
    --good_json /path/to/good_labels.json
```
Saves `best_model.pt` to the same directory as `good_json`.

**RL Agent (runs locally, ~10 min):**
```bash
python src/train_rl.py \
    --save_dir models/ \
    --episodes 500 \
    --grid_size 15
```
Saves `dqn_rover.pt` to `models/`.

## Usage

```python
import torch
from src.perception import TerrainClassifier

# Load perception model
tc = TerrainClassifier()
model = tc.get_model()
model.load_state_dict(torch.load('models/best_model.pt', map_location='cpu'))
model.eval()

# Load RL agent
from src.rl_agent import DoubleDQN
agent = DoubleDQN()
agent.online_net.load_state_dict(torch.load('models/dqn_rover.pt', map_location='cpu'))
agent.epsilon = 0.05
```
