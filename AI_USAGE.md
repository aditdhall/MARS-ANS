# AI Usage Log — ANS Mars Rover Navigation System

**Course:** AI 710 — Principles of Machine Learning
**Team:** Adit Dhall · Matthew Landon · Thejas Nagesh Gowda

---

## Overview

This document logs how AI assistance (Claude by Anthropic) was used throughout the development of the ANS system, as required by the course guidelines.

---

## AI Tools Used

- **Claude (Anthropic)** — Primary AI assistant for code generation, architecture design, and debugging
- **GitHub Copilot** — Minor autocomplete suggestions during development

---

## Detailed Usage Log

### 1. Architecture Design
**What AI helped with:** Discussing the overall system architecture — how to connect perception, sensor fusion, planning, and RL into a coherent pipeline. Claude helped us think through the tradeoffs between different approaches (e.g. patch classifier vs segmentation head, MC Dropout vs Deep Ensembles, A* vs Theta* vs D* Lite).

**What we decided ourselves:** The specific research questions, the H-score formulation connecting uncertainty to planning, the decision to add camera-LiDAR sensor fusion as a contribution beyond the original paper design.

**Reflection:** AI was useful for rapidly evaluating options, but the core design decisions — especially what makes our system novel vs prior work — required our own understanding of the literature we reviewed.

---

### 2. Data Pipeline (data.py)
**What AI helped with:** Generating the `AI4MarsDataset` class structure and `get_dataloaders` function. Fixing the label resize bug (using `Image.NEAREST` to preserve class IDs). Adding `num_workers=4` and `pin_memory=True` for performance.

**What we verified ourselves:** Ran verification cells to confirm image/label shapes, unique label values, and class distribution. Visually inspected sample images overlaid with labels to confirm alignment.

**Reflection:** The label resize bug was caught by Claude before we ran training — without that fix, training would have produced silently wrong results. This shows AI is useful for code review, not just generation.

---

### 3. Perception Module (perception.py)
**What AI helped with:** MobileNetV3-Small fine-tuning setup, MC Dropout inference loop (keeping dropout active during eval), entropy computation formula, geometry layer base values per terrain class, sensor fusion and H-score functions.

**What we verified ourselves:** Ran unit tests on all three new functions to confirm geometry values matched expected physical intuition (big_rocks highest roughness, bedrock lowest). Confirmed entropy values are higher for uncertain predictions.

**Reflection:** The MC Dropout implementation required understanding that `model.eval()` disables dropout, so we needed to selectively re-enable it. This is a subtle PyTorch behavior that AI explained clearly.

---

### 4. Training Loop (train.py)
**What AI helped with:** Basic training loop structure, the `get_dominant_label` function to convert pixel-level labels to image-level labels for classification, argparse setup, saving best model to Drive.

**What we verified ourselves:** Monitored training curves across 30 epochs. Identified overfitting after epoch 2 (val loss rising while train loss falling) and confirmed best_model.pt was saved at epoch 2. Final val accuracy: 92.83%.

**Reflection:** The patch classifier approach (dominant label per image) was a deliberate simplification from pixel-level segmentation. AI helped us see this was appropriate for our scope and that the trained model would still produce meaningful uncertainty estimates via MC Dropout.

---

### 5. Planners (planner.py)
**What AI helped with:** Theta* implementation with line-of-sight checks using Bresenham's algorithm. D* Lite implementation with incremental replanning. Both used as reference implementations that we verified and tested.

**What we verified ourselves:** Tested on a 10x10 grid with a wall. Confirmed Theta* produces shorter any-angle paths vs D* Lite's grid-constrained paths. Ran 20-trial comparison showing Theta* path length=2 vs D* Lite=15 on 15x15 grids.

**Reflection:** True D* Lite incremental updating is complex. Our implementation does full replanning after cell updates, which is a known simplification. AI was transparent about this limitation and we documented it in the paper.

---

### 6. RL Agent (rl_agent.py)
**What AI helped with:** MarsRoverEnv class with continuous position, 8-directional action space, and H-score based reward function. Double DQN network architecture, replay buffer, soft target update.

**What we verified ourselves:** Ran 5 test episodes to confirm environment step/reset mechanics work. Confirmed epsilon decay schedule reaches target value after 500 episodes. Verified mission completion after full training.

**Reflection:** The reward function design — specifically how to weight H-score penalty vs progress reward vs goal reward — required multiple iterations. The final values (-1 step, -5*H_score, +10*progress, +100 goal) were tuned based on observing agent behavior, not generated directly by AI.

---

### 7. Dashboard (dashboard.py)
**What AI helped with:** Streamlit layout, loading pre-trained weights, progress bar during training, metric display.

**What we verified ourselves:** Confirmed dashboard produces different random maps on each run. Confirmed loaded RL agent completes missions (completed=True, 0 collisions).

---

### 8. Ablation Experiments
**What AI helped with:** run_ablation function structure, result collection and JSON saving, summary table formatting.

**What we analyzed ourselves:** Interpreted the results — alpha=0.75 optimal because uncertainty signal must dominate but terrain volatility still contributes. H_crit=0.7 optimal because 0.4 is too strict (blocks navigation) and 0.8 is too permissive (dangerous terrain). These interpretations are our own analysis.

---

## What AI Did NOT Do

- AI did not write the literature review or the paper text
- AI did not select which papers to review or determine their relevance
- AI did not interpret experimental results — all interpretation is our own
- AI did not make architectural decisions about what makes our system novel
- AI did not run any experiments — all training and evaluation was run by us

---

## Reflection on AI-Assisted Development

Using Claude as a coding assistant significantly accelerated development but also required careful verification at every step. Key lessons:

1. **AI generates plausible-looking code that may have subtle bugs.** The label resize bug and the GOOD_JSON corruption issue would have caused silent failures without our verification.

2. **AI is most useful for boilerplate and standard patterns.** The DataLoader setup, training loop, and Streamlit layout are standard patterns that AI generated correctly. Novel contributions (H-score design, reward function tuning) required our own judgment.

3. **AI explanations are valuable for learning.** Understanding why MC Dropout requires selective re-enabling of dropout layers, or why Double DQN reduces overestimation, deepened our understanding beyond just getting working code.

4. **Human oversight is essential.** Every major component was tested with unit tests and sanity checks before being integrated. We never blindly trusted AI-generated code.
