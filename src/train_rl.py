import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import config
from perception import compute_hscore, get_geometry
from rl_agent import DoubleDQN, MarsRoverEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--grid_size", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Training Double DQN for {args.episodes} episodes on {args.grid_size}x{args.grid_size} grid")
    print("Grid regenerated every 100 episodes for robustness")
    print("-" * 60)

    H = W = args.grid_size
    classes = ["soil", "bedrock", "sand", "big_rocks"]

    def build_cost_map():
        terrain_grid = [[random.choice(classes) for _ in range(W)] for _ in range(H)]
        cost_map = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                cls  = terrain_grid[i][j]
                geom = get_geometry(cls)
                h    = compute_hscore(geom["lidar_conf"] * 0.3, geom, alpha=config.ALPHA)
                if h > config.H_CRIT:
                    cost_map[i, j] = 999
                else:
                    cost_map[i, j] = (
                        0.3 * config.TERRAIN_COST[cls]
                        + 0.2 * geom["slope"]
                        + 0.2 * geom["roughness"]
                        + 0.3 * h
                    )
        cost_map[0, 0]   = 0.15
        cost_map[-1, -1] = 0.15
        return cost_map

    cost_map = build_cost_map()
    start    = (0, 0)
    goal     = (H - 1, W - 1)
    env      = MarsRoverEnv(cost_map, start, goal)
    agent    = DoubleDQN()

    best_reward = float("-inf")
    save_path   = os.path.join(args.save_dir, "dqn_rover.pt")
    rewards_log = []

    for ep in range(1, args.episodes + 1):
        if ep % 100 == 1:
            cost_map = build_cost_map()
            env      = MarsRoverEnv(cost_map, start, goal)

        state        = env.reset()
        total_reward = 0.0
        done         = False
        steps        = 0

        while not done and steps < 300:
            action                   = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train_step()
            state         = next_state
            total_reward += reward
            steps        += 1

        agent.update_epsilon()
        rewards_log.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.online_net.state_dict(), save_path)

        if ep % 100 == 0:
            avg = sum(rewards_log[-100:]) / 100
            print(f"Episode {ep:4d}: reward={total_reward:8.2f}  avg_100={avg:8.2f}  epsilon={agent.epsilon:.3f}  steps={steps}")

    torch.save(agent.online_net.state_dict(),
               os.path.join(args.save_dir, "dqn_rover_final.pt"))

    print("-" * 60)
    print(f"Training complete. Best reward: {best_reward:.2f}  Final epsilon: {agent.epsilon:.3f}")
