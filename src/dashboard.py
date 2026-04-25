import os
import sys
import random

import numpy as np
import streamlit as st
import torch

# Add src to path so imports work when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from evaluate import run_episode, visualize_result
from planner import ThetaStar
from rl_agent import DoubleDQN, MarsRoverEnv

# Paths relative to project root
ROOT       = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, 'models')
FIGURES_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

st.title("ANS Mars Rover Navigation System")
st.caption("AI 710 · Rochester Institute of Technology · Spring 2026")

alpha    = st.sidebar.slider("alpha (H-score weight)", 0.0, 1.0, 0.75, 0.05)
h_crit   = st.sidebar.slider("h_crit (impassability threshold)", 0.3, 0.9, 0.7, 0.05)
episodes = st.sidebar.slider("RL episodes to train", 10, 200, 50, 10)
run      = st.sidebar.button("🚀 Run Mission")

if run:
    import config
    from perception import get_geometry, compute_hscore

    classes = ["soil", "bedrock", "sand", "big_rocks"]
    H = W = 15

    st.write("Building cost map...")
    cost_map = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            cls  = random.choice(classes)
            geom = get_geometry(cls)
            h    = compute_hscore(geom['lidar_conf'] * 0.3, cls, alpha)
            if h > h_crit:
                cost_map[i, j] = 999
            else:
                cost_map[i, j] = (0.3 * config.TERRAIN_COST[cls]
                                 + 0.2 * geom['slope']
                                 + 0.2 * geom['roughness']
                                 + 0.3 * h)
    cost_map[0, 0]   = 0.15
    cost_map[H-1, W-1] = 0.15

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cost_map, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Cost Map (α={alpha}, Hcrit={h_crit})")
    st.pyplot(fig)

    start, goal = (0, 0), (H - 1, W - 1)
    planner = ThetaStar(cost_map)
    path    = planner.find_path(start, goal)
    st.write(f"Theta* global path: {len(path)} waypoints")

    env   = MarsRoverEnv(cost_map, start, goal)
    agent = DoubleDQN()

    dqn_path = os.path.join(MODELS_DIR, 'dqn_rover.pt')
    if os.path.exists(dqn_path):
        agent.online_net.load_state_dict(torch.load(dqn_path, map_location='cpu'))
        agent.epsilon = 0.05
        st.write("Loaded pre-trained RL agent ✓")
    else:
        st.write(f"No pre-trained agent found — training for {episodes} episodes...")
        progress = st.progress(0)
        for ep in range(episodes):
            state = env.reset()
            done, steps = False, 0
            while not done and steps < 300:
                action          = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.store(state, action, reward, next_state, done)
                state = next_state
                agent.train_step()
                steps += 1
            agent.update_epsilon()
            progress.progress((ep + 1) / episodes)

    st.write("Running final mission episode...")
    result = run_episode(env, agent, planner)

    out_path = os.path.join(FIGURES_DIR, 'result.png')
    visualize_result(cost_map, result["trajectory"], start, goal,
                     title="Final Mission", save_path=out_path)
    st.image(out_path)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Completed",     str(result["completed"]))
    col2.metric("Steps",         result["steps"])
    col3.metric("Total Reward",  f"{result['total_reward']:.1f}")
    col4.metric("Collisions",    result["collisions"])
