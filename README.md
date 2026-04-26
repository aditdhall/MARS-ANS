# ANS: AI Autonomous Navigation System for Mars Rovers

**AI 710 — Principles of Machine Learning**
**Rochester Institute of Technology — Spring 2026**
**Team:** Adit Dhall · Matthew Landon · Thejas Nagesh Gowda

---

## Overview

ANS is an end-to-end autonomous navigation system for Mars rovers during communication blackout periods. It integrates terrain perception, sensor fusion, uncertainty-aware planning, and reinforcement learning into a single deployable pipeline.

**Key results:**
- MobileNetV3-Small terrain classifier: **92.83% validation accuracy** on AI4Mars
- MC Dropout uncertainty estimation with T=30 stochastic passes
- Camera + simulated LiDAR sensor fusion via H-score
- Theta* any-angle planner + D* Lite incremental replanner
- Double DQN local navigator: **100% mission completion** (alpha=0.75, Hcrit=0.7)

---

## Repository Structure

```
MARS-ANS/
├── src/                    # All source code
│   ├── config.py           # Hyperparameters and constants
│   ├── data.py             # AI4Mars dataset loader + cost map builder
│   ├── perception.py       # MobileNetV3, MC Dropout, sensor fusion, H-score
│   ├── planner.py          # Theta* and D* Lite planners
│   ├── rl_agent.py         # MarsRoverEnv + Double DQN
│   ├── evaluate.py         # Episode runner, visualization, ablation
│   ├── train.py            # MobileNetV3 training script
│   ├── train_rl.py         # RL agent training script
│   ├── run_ablations.py    # Ablation sweep runner
│   ├── dashboard.py        # Streamlit interactive dashboard
│   ├── simulation.py       # Live pygame simulation demo
│   └── inference_server.py # Flask inference server for cluster deployment
├── notebooks/
│   └── ANS_Perception.ipynb  # Perception pipeline and MC Dropout analysis
├── models/
│   └── README.md           # Instructions for downloading model weights
├── configs/
│   └── default.yaml        # All hyperparameters in YAML format
├── data/
│   └── README.md           # Instructions for obtaining AI4Mars dataset
├── figures/                # Generated plots and visualizations
├── requirements.txt        # Python dependencies
├── AI_USAGE.md             # AI assistance log
└── README.md               # This file
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/aditdhall/MARS-ANS.git
cd MARS-ANS
```

### 2. Set up conda environment
```bash
conda activate mars_ans
pip install -r requirements.txt
```

> Alternatively, recreate the environment from scratch:
> ```bash
> conda env create -f environment.yml
> conda activate mars_ans
> ```

### 3. Download model weights
See `models/README.md` for instructions on downloading pre-trained weights.

### 4. Get the dataset
See `data/README.md` for instructions on obtaining AI4Mars.

---

## Running the System

### Interactive Dashboard
```bash
cd src
streamlit run dashboard.py
```

### Train MobileNetV3 (Colab recommended)
```bash
python src/train.py \
    --img_dir /path/to/ai4mars/images/edr \
    --label_dir /path/to/ai4mars/labels/train \
    --good_json /path/to/good_labels.json
```

### Train RL Agent
```bash
python src/train_rl.py \
    --save_dir models/ \
    --episodes 500 \
    --grid_size 15
```

### Run Ablation Experiments
```bash
cd src
python run_ablations.py
```

### Live Pygame Simulation

The simulation requires two terminals: one on the cluster running the inference server,
and one local running the pygame window.

**Terminal 1 — Cluster (inference server):**
```bash
ssh ad6449@narnia.gccis.rit.edu
tmux attach -t mars
conda activate mars_ans
cd ~/MARS-ANS
export CUDA_VISIBLE_DEVICES=4
python3 src/inference_server.py
```
Detach from tmux with `Ctrl+B, D` — the server keeps running in the background.

**Terminal 2 — Local (SSH tunnel, keep open):**
```bash
ssh -L 5000:localhost:5000 ad6449@narnia.gccis.rit.edu -N
```

**Terminal 3 — Local (pygame window):**
```bash
python src/simulation.py
```

The simulation will print `Inference server: ONLINE` if the Flask server is reachable,
or fall back to simulated CNN probabilities if not.

**Controls:**
| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `R` | Reset with new random map |
| `Q` / `ESC` | Quit |

### Full Colab Notebook
Upload `notebooks/ANS_Perception.ipynb` to Google Colab. Run all cells in order.

---

## System Architecture

```
AI4Mars Images → MobileNetV3-Small → Terrain Class + Confidence
                      ↓ MC Dropout (T=30)
                 Uncertainty Map U(x,y)
                      +
Simulated LiDAR → Slope + Roughness + LiDAR Confidence
                      ↓ Sensor Fusion
                 U_fused = β·U_mcdropout + (1-β)·|cam_conf - lidar_conf|
                      ↓ H-Score
                 H(x,y) = α·U_fused + (1-α)·V(x,y)
                      ↓ Cost Map
                 C(x,y) = wt·T + ws·S + wr·R + wh·H
                      ↓
         Theta* Global Planner → Waypoints
                      ↓
         Double DQN Local Navigator → Continuous Actions
```

The ANS system uses a three-layer navigation architecture:
- **Theta*** computes the globally optimal reference path at mission start (shown in blue)
- **RL agent** (Double DQN) drives the rover reactively step by step
- **D* Lite** overrides the RL agent when perception conflict is detected

---

## Key Results

### Alpha Sweep Results (H_crit=0.8 fixed, 50 episodes each)
| Alpha | Completion | Avg Reward | Avg Collisions |
|-------|-----------|------------|----------------|
| 0.00  | 0%        | 8.39       | 1.00           |
| 0.25  | 92%       | 222.39     | 0.08           |
| 0.50  | 92%       | 225.33     | 0.08           |
| 0.75  | 90%       | 218.19     | 0.10           |
| **1.00**  | **94%**   | **229.57** | **0.06**   |

### H_crit Sweep Results (Alpha=1.0 fixed, 50 episodes each)
| H_crit | Completion | Avg Reward | Avg Collisions |
|--------|-----------|------------|----------------|
| 0.4    | 92%       | 226.27     | 0.08           |
| 0.5    | 90%       | 217.30     | 0.10           |
| 0.6    | 90%       | 217.70     | 0.10           |
| 0.7    | 92%       | 222.94     | 0.08           |
| **0.8**    | **94%** | **230.67** | **0.06**   |

### Planner Comparison
| Planner  | Avg Path Length | Avg Time (ms) | Completion |
|----------|----------------|---------------|------------|
| Theta*   | 2.00           | 0.252         | 20/20      |
| D* Lite  | 15.00          | 0.047         | 20/20      |

---

## References

See paper for full citation list. Key references:
- Swan et al. (2021) — AI4Mars dataset
- Gal & Ghahramani (2016) — MC Dropout
- Koenig & Likhachev (2002) — D* Lite
- Cai et al. (2024) — EVORA risk-aware navigation
