# AI Usage Log — ANS Mars Rover Navigation System

**Course:** AI 710 — Principles of Machine Learning  
**Team:** Adit Dhall · Matthew Landon · Thejas Nagesh Gowda

---

## Overview

This file documents how we used AI tools during the project. The pattern was pretty consistent throughout: we designed the architecture, worked out the math, made the research decisions, and then gave Claude a detailed spec to implement in Python. Claude wrote a lot of the code. The ideas and design were ours.

---

## AI Tools Used

- **Claude (Anthropic)** — main tool, used for almost all code generation and debugging
- **GitHub Copilot** — occasional autocomplete, nothing significant

---

## Detailed Usage Log

---

### 1. Architecture Design

**Tool:** Claude

**Prompt/Request:** We described the four-stage pipeline we designed (perception → sensor fusion → planning → RL), the H-score formulation `H(x,y) = α·U_fused + (1-α)·V(x,y)` and the sensor fusion equation `U_fused = β·U_mcdropout + (1-β)|cam_conf - lidar_conf|`. We asked Claude to help think through implementation tradeoffs — patch classifier vs segmentation head, MC Dropout vs Deep Ensembles, A* vs Theta* vs D* Lite.

**What was generated:** Discussion of tradeoffs, pros/cons of each approach, suggested implementation order for the modules.

**Modifications we made:** All actual design decisions were done by us, whre it includes the H-score formulation, the sensor fusion equation, the two-planner design, the choice of MobileNetV3-Small. Claude didn't suggest any of these, it just helped us think through implementation once we'd already decided.

**What we learned:** AI is useful for stress-testing a design which we already made, but it can't tell you what's novel or what the right research question is. That part has to come from reading the literature yourself.

---

### 2. Data Pipeline (data.py)

**Tool:** Claude

**Prompt/Request:** "Implement a PyTorch Dataset and DataLoader for the AI4Mars dataset. Use an 80/10/10 train/val/test split, class weights [2.7, 2.0, 8.3, 111.0] and data augmentation with horizontal flip and color jitter."

**What was generated:** The full `AI4MarsDataset` class and `get_dataloaders` function. Claude also flagged during generation that using bilinear interpolation to resize label masks would corrupt class IDs, it suggested using `Image.NEAREST` instead.

**Modifications we made:** The class weights and split ratios were computed by us from actual dataset class frequencies before we wrote the prompt. We also added verification cells ourselves to confirm image/label shapes, unique label values and visual alignment.

**What we learned:** The label resize bug was a real catch, bilinear interpolation on a segmentation mask creates interpolated pixel values that don't correspond to any real class. Without that fix, training would've produced silently wrong results. Good reminder that code review from AI can catch things generation alone wouldn't.

---

### 3. Perception Module (perception.py)

**Tool:** Claude

**Prompt/Request:** "Implement the MC Dropout inference loop with T=30 passes, the sensor fusion function and the H-score function using these exact formulas and terrain geometry values: [we listed slope, roughness, lidar_conf, and volatility per class]."

**What was generated:** `run_mc_dropout`, `compute_fusion`, `compute_hscore`, and `get_geometry` functions.

**Modifications we made:** The geometry base values per terrain class came entirely from us, where we worked those out from physical intuition about Mars terrain (big rocks = high roughness and slope, bedrock = smooth and stable, etc.). Claude flagged during implementation that `model.eval()` disables dropout, so we needed to selectively re-enable dropout layers during MC Dropout passes. We verified this behavior ourselves before accepting the code.

**What we learned:** MC Dropout has a subtle PyTorch gotcha, eval mode disables dropout, which defeats the whole point. Claude flagged it clearly and explained why. Understanding it properly meant we could verify the implementation was correct rather than just trusting it.

---

### 4. Training Loop (train.py)

**Tool:** Claude

**Prompt/Request:** "Implement a PyTorch training loop for 30 epochs with Adam at lr=1e-4, class-weighted CrossEntropyLoss, validation tracking each epoch, and saving the best checkpoint by val loss."

**What was generated:** The full training loop, `get_dominant_label` function to convert pixel-level masks to image-level labels, and an argparse CLI.

**Modifications we made:** The decision to use dominant-label classification instead of pixel-level segmentation was ours, this was a deliberate simplification to keep scope manageable while still getting meaningful uncertainty estimates from MC Dropout. We monitored training curves ourselves across 30 epochs and identified that epoch 2 was the best checkpoint (val loss started climbing after that).

**What we learned:** The patch classifier simplification works well for this use case. Final val accuracy: 92.83%. Watching the training curves yourself is important and you can't just trust that "best checkpoint" means the model is actually good without understanding what's happening.

---

### 5. Planners (planner.py)

**Tool:** Claude

**Prompt/Request:** "Implement two planners: (1) Theta* for global any-angle path planning using Bresenham's line-of-sight check and (2) D* Lite that replans from the rover's current position when triggered. Both should operate on a grid cost map."

**What was generated:** `ThetaStar` and `DStarLite` classes.

**Modifications we made:** The two-planner design was ours, where Theta* for global planning, D* Lite for reactive replanning when H-score spikes. We tested both on a 10×10 grid with a wall and ran a 20-trial comparison ourselves: Theta* avg path length = 2, D* Lite = 15, which confirmed any-angle paths are significantly shorter.

**What we learned:** Our D* Lite does full A* replanning on each update rather than true incremental updating. Claude was upfront about this being a simplification. We understood the tradeoff, documented it in the paper and decided it was acceptable for our use case.

---

### 6. RL Agent (rl_agent.py)

**Tool:** Claude

**Prompt/Request:** "Implement MarsRoverEnv with continuous position, 8-directional actions at 0.5-cell steps, reward function r = -1 - 5·H_score + 10·(prev_dist - new_dist) with +100 goal bonus, and a 6-dimensional state vector and also implement Double DQN with replay buffer size 10000, soft target update τ=0.005, γ=0.99, lr=1e-4, batch=64, ε decay=0.995."

**What was generated:** `MarsRoverEnv` and `DoubleDQN` classes.

**Modifications we made:** Every value in that prompt came from us. The reward weights (-1, -5, +10, +100) were tuned over multiple training runs by watching agent behavior. We ran 5 test episodes ourselves to confirm step/reset mechanics, verified epsilon decay and confirmed 100% mission completion after full training.

**What we learned:** Reward function design is genuinely hard and AI can't do it for you, it requires watching the agent actually behave and iterating. The H-score penalty in the reward is what ties perception uncertainty directly to navigation decisions, which is the core idea of the whole system.

---

### 7. Inference Server (inference_server.py)

**Tool:** Claude

**Prompt/Request:** "Implement a Flask server with /health, /infer and /batch_infer endpoints that loads our trained MobileNetV3 model and runs MC Dropout inference on request. It needs to serve predictions over HTTP so a local pygame simulation can get real CNN outputs from Narnia."

**What was generated:** The full Flask server and class index builder.

**Modifications we made:** The endpoint design and the architecture (local pygame + Narnia GPU server connected over SSH tunnel) were our decisions. We tested all three endpoints and confirmed the simulation correctly prints `Inference server: ONLINE` and receives real probability distributions.

**What we learned:** Separating inference onto Narnia and rendering locally over a tunnel is a clean pattern for GPU-heavy demos. Worth keeping in mind for future projects.

---

### 8. Live Simulation (simulation.py)

**Tool:** Claude

**Prompt/Request:** "Implement a pygame simulation with a 2×3 panel layout: navigation map, CNN bar chart, LiDAR geometry panel, terrain image, camera sensor view, LiDAR sensor view. D* Lite replanning triggers if CNN class mismatches expected terrain OR H-score > 0.65. Show fading replan history on the map."

**What was generated:** The full pygame application by drawing all functions, the rover step loop and the conflict/replan logic.

**Modifications we made:** The panel layout, the conflict detection thresholds and the fading history idea were all ours. We ran the simulation end-to-end with the inference server live on Narnia, confirmed all six panels update correctly and verified replanning triggers at the right moments.

**What we learned:** Designing the visual layout before implementing it makes Claude's output much cleaner. When we gave it a precise grid spec, it got it right first try.

---

### 9. Ablation Experiments (run_ablations.py, evaluate.py)

**Tool:** Claude

**Prompt/Request:** "Implement an ablation runner that sweeps alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0} and H_crit ∈ {0.4, 0.5, 0.6, 0.7, 0.8}, runs 20 episodes per configuration and tracks completion rate, avg steps, avg reward, and avg collisions. Save results to JSON."

**What was generated:** `run_ablation` and `run_episode` functions and JSON output.

**Modifications we made:** We designed the sweep ranges and chose which metrics to track. We interpreted all results ourselves — alpha=0.75 is optimal because uncertainty needs to dominate but terrain volatility still contributes. H_crit=0.7 hits the right balance between blocking dangerous terrain and not blocking everything.

**What we learned:** Ablation design requires domain knowledge to be meaningful. Choosing which parameters to sweep and what range to sweep them over came from understanding what the H-score is actually doing, not from running code.

---

## What Claude Did NOT Do

- Did not design the H-score formula or sensor fusion equation
- Did not choose which algorithms to use, all architecture decisions were ours
- Did not set any hyperparameters
- Did not write the paper or literature review
- Did not interpret experimental results
- Did not run training or experiments
- Did not decide what makes this system novel

---

## Reflection

**How did AI tools help productivity?**
Massively, on the implementation side. Writing boilerplate PyTorch classes, Flask servers, and pygame drawing functions from scratch would've taken a lot longer. Claude let us spend most of our time on the things that actually mattered — designing the system, tuning it and analyzing results.

**What were they not helpful for?**
Anything that required actually understanding the problem. The reward function weights, the H-score alpha, the terrain geometry values, the ablation sweep ranges, all of that required domain knowledge that Claude doesn't have. When we asked vague questions, we got vague answers. The design work had to come from us.

**How did we verify AI-generated code was correct?**
Every module got unit tested before integration. We checked data shapes, printed intermediate values, ran sanity checks on outputs and compared against expected physical intuition (e.g. big rocks should have the highest roughness value). We also ran the full pipeline end-to-end multiple times. We never just trusted the output and moved on, if we couldn't explain why the code worked, we didn't accept it.
