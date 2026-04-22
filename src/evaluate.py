import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from perception import compute_hscore, get_geometry
from rl_agent import DoubleDQN, MarsRoverEnv


def run_episode(env, agent, planner, max_steps=500):
    waypoints = planner.find_path(
        (int(env.start[0]), int(env.start[1])),
        (int(env.goal[0]),  int(env.goal[1])),
    )
    state      = env.reset()
    trajectory = [(env.current_pos[0], env.current_pos[1])]
    rewards    = []
    total_reward = 0.0
    collisions   = 0
    completed    = False
    steps        = 0

    for _ in range(max_steps):
        action               = agent.select_action(state)
        next_state, reward, done = env.step(action)
        total_reward        += reward
        rewards.append(reward)
        trajectory.append((env.current_pos[0], env.current_pos[1]))
        steps += 1
        if done:
            if reward <= -10.0:
                collisions += 1
            else:
                completed = True
            break
        state = next_state

    return {
        "completed":    completed,
        "steps":        steps,
        "total_reward": float(total_reward),
        "trajectory":   trajectory,
        "collisions":   collisions,
        "waypoints":    waypoints,
        "rewards":      rewards,
    }


def visualize_result(cost_map, trajectory, start, goal,
                     title="Mission", save_path="figures/result.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(cost_map, cmap="viridis")
    if trajectory:
        axes[0].plot([p[1] for p in trajectory], [p[0] for p in trajectory],
                     color="cyan", linewidth=2, label="trajectory")
    axes[0].scatter([start[1]], [start[0]], color="lime", s=100, zorder=5, label="start")
    axes[0].scatter([goal[1]],  [goal[0]],  color="red",  s=100, zorder=5, label="goal")
    axes[0].set_title(f"{title} — cost map")
    axes[0].legend()

    dists       = [math.hypot(p[0]-goal[0], p[1]-goal[1]) for p in trajectory]
    step_reward = [-d for d in dists]
    axes[1].plot(range(len(step_reward)), step_reward, color="purple")
    axes[1].set_title("Per-step reward (neg. distance to goal)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("reward")
    axes[1].grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def run_ablation(n_runs=20, alpha=0.5, h_crit=0.6, grid_size=15,
                 dqn_path="models/dqn_rover.pt"):
    classes    = ["soil", "bedrock", "sand", "big_rocks"]
    completions, all_steps, all_rewards, all_collisions = [], [], [], []
    H = W = grid_size

    for run in range(1, n_runs + 1):
        cost_map = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                cls  = random.choice(classes)
                geom = get_geometry(cls)
                h    = compute_hscore(geom["lidar_conf"] * 0.3, cls, alpha)
                if h > h_crit:
                    cost_map[i, j] = 999
                else:
                    cost_map[i, j] = (0.3 * config.TERRAIN_COST[cls]
                                     + 0.2 * geom["slope"]
                                     + 0.2 * geom["roughness"]
                                     + 0.3 * h)
        cost_map[0, 0]   = 0.15
        cost_map[-1, -1] = 0.15

        env   = MarsRoverEnv(cost_map, (0, 0), (H-1, W-1))
        agent = DoubleDQN()
        if os.path.exists(dqn_path):
            agent.online_net.load_state_dict(torch.load(dqn_path, map_location="cpu"))
            agent.epsilon = 0.05

        state                            = env.reset()
        total_reward, steps, collisions  = 0.0, 0, 0
        completed, done                  = False, False
        while not done and steps < 500:
            action          = agent.select_action(state)
            state, reward, done = env.step(action)
            total_reward   += reward
            steps          += 1
            if done:
                if reward <= -10.0:
                    collisions += 1
                else:
                    completed = True

        completions.append(completed)
        all_steps.append(steps)
        all_rewards.append(total_reward)
        all_collisions.append(collisions)
        if run % 5 == 0:
            print(f"  Run {run}/{n_runs} done")

    return {
        "completion_rate": float(np.mean(completions)),
        "avg_steps":       float(np.mean(all_steps)),
        "avg_reward":      float(np.mean(all_rewards)),
        "avg_collisions":  float(np.mean(all_collisions)),
    }
