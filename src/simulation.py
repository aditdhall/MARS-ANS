"""
ANS Mars Rover — Live Simulation Demo
======================================
Pygame window with 2 rows and 6 panels showing the rover navigating
in real time with live sensor readings.

Controls:
  SPACE  — pause / resume
  R      — reset with new random map
  Q/ESC  — quit

Run from project root:
  cd src && python simulation.py
"""

import math
import os
import random
import sys
import time

import numpy as np
import pygame

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

import config
from perception import get_geometry, compute_fusion, compute_hscore
from planner import ThetaStar
from rl_agent import MarsRoverEnv, DoubleDQN
import torch

# ─── Layout constants ─────────────────────────────────────────────────────────
W_SCREEN   = 1280
H_SCREEN   = 780
H_ROW      = 330
H_STATUS   = 60
W_LEFT     = 460
W_MID      = 360
W_RIGHT    = W_SCREEN - W_LEFT - W_MID   # 460

PANEL_PAD  = 10
FPS        = 60
STEP_DELAY = 0.8  # seconds between rover steps

# ─── Colours ──────────────────────────────────────────────────────────────────
BG          = (18, 18, 28)
PANEL_BG    = (28, 28, 42)
PANEL_EDGE  = (60, 60, 90)
WHITE       = (255, 255, 255)
GREY        = (160, 160, 180)
DARK_GREY   = (80,  80, 100)
GREEN       = (50,  220, 100)
RED         = (220,  60,  60)
ORANGE      = (240, 160,  40)
CYAN        = (40,  220, 220)
YELLOW      = (240, 220,  60)
PURPLE      = (180,  80, 220)
LIME        = (120, 240,  60)

TERRAIN_COLORS = {
    "soil":      (176, 142,  97),
    "bedrock":   (110, 110, 130),
    "sand":      (210, 195, 140),
    "big_rocks": (130,  80,  50),
}
TERRAIN_NAMES = ["soil", "bedrock", "sand", "big_rocks"]

# ─── Synthetic terrain images (generated, no real images needed) ──────────────
def make_terrain_surface(cls, size=120):
    """Generate a synthetic Mars-like terrain image for each class."""
    surf = pygame.Surface((size, size))
    base = TERRAIN_COLORS[cls]
    rng  = np.random.RandomState(hash(cls) % 2**31)

    for x in range(size):
        for y in range(size):
            noise = rng.randint(-25, 25)
            r = max(0, min(255, base[0] + noise))
            g = max(0, min(255, base[1] + noise // 2))
            b = max(0, min(255, base[2] + noise // 3))
            surf.set_at((x, y), (r, g, b))

    if cls == "big_rocks":
        for _ in range(8):
            rx, ry = rng.randint(5, size-15), rng.randint(5, size-15)
            rr = rng.randint(6, 18)
            pygame.draw.ellipse(surf, (80, 55, 35), (rx, ry, rr*2, rr))
            pygame.draw.ellipse(surf, (60, 40, 25), (rx+2, ry+2, rr*2-4, rr-4))
    elif cls == "sand":
        for _ in range(5):
            sx = rng.randint(0, size-20)
            pygame.draw.arc(surf, (190, 175, 120),
                            (sx, rng.randint(20, size-20), rng.randint(20,50), 8),
                            0, math.pi, 2)
    elif cls == "bedrock":
        for _ in range(6):
            x1, y1 = rng.randint(0, size), rng.randint(0, size)
            x2, y2 = x1 + rng.randint(-30, 30), y1 + rng.randint(-30, 30)
            pygame.draw.line(surf, (85, 85, 100), (x1, y1), (x2, y2), 1)

    return surf

def load_terrain_images(img_dir):
    """Load real terrain JPGs from img_dir, falling back to synthetic ones on failure."""
    surfs   = {}
    loaded  = []
    for cls in TERRAIN_NAMES:
        path = os.path.join(img_dir, f"{cls}.JPG")
        try:
            surf = pygame.image.load(path).convert()
            surfs[cls] = surf
            loaded.append(cls)
        except (pygame.error, FileNotFoundError) as e:
            print(f"  [{cls}] failed to load {path}: {e} — using synthetic fallback")
            surfs[cls] = make_terrain_surface(cls)
    if loaded:
        print(f"Loaded real terrain images: {', '.join(loaded)}")
    else:
        print("No real terrain images loaded — all synthetic")
    return surfs


# ─── Helper drawing functions ──────────────────────────────────────────────────
def draw_panel(surf, rect, title=None, edge_color=PANEL_EDGE):
    pygame.draw.rect(surf, PANEL_BG, rect, border_radius=8)
    pygame.draw.rect(surf, edge_color, rect, 2, border_radius=8)
    if title:
        font_sm = pygame.font.SysFont("monospace", 13, bold=True)
        label   = font_sm.render(title, True, GREY)
        surf.blit(label, (rect.x + PANEL_PAD, rect.y + PANEL_PAD))


def draw_bar_chart(surf, rect, values, labels, colors, title=""):
    draw_panel(surf, rect)
    font_sm = pygame.font.SysFont("monospace", 11)
    font_ti = pygame.font.SysFont("monospace", 13, bold=True)
    if title:
        surf.blit(font_ti.render(title, True, GREY), (rect.x+PANEL_PAD, rect.y+PANEL_PAD))

    n     = len(values)
    inner = rect.inflate(-20, -40)
    inner.y += 24
    bw    = inner.width // n - 4
    max_h = inner.height - 24

    for i, (val, lbl, col) in enumerate(zip(values, labels, colors)):
        bh = int(val * max_h)
        bx = inner.x + i * (bw + 4)
        by = inner.y + max_h - bh
        pygame.draw.rect(surf, col, (bx, by, bw, bh), border_radius=3)
        pct = font_sm.render(f"{val:.2f}", True, WHITE)
        surf.blit(pct, (bx, by - 14))
        lbl_s = font_sm.render(lbl[:4], True, GREY)
        surf.blit(lbl_s, (bx, inner.y + max_h + 2))


def draw_radar(surf, rect, values, labels, color, title=""):
    """Draw a radar/spider chart."""
    draw_panel(surf, rect)
    font_sm = pygame.font.SysFont("monospace", 11)
    font_ti = pygame.font.SysFont("monospace", 13, bold=True)
    if title:
        surf.blit(font_ti.render(title, True, GREY), (rect.x+PANEL_PAD, rect.y+PANEL_PAD))

    cx    = rect.centerx
    cy    = rect.centery + 10
    r_max = min(rect.width, rect.height) // 2 - 28
    n     = len(values)

    # Draw grid rings
    for ring in [0.25, 0.5, 0.75, 1.0]:
        pts = []
        for i in range(n):
            angle = math.pi/2 + 2*math.pi*i/n
            px = cx + ring * r_max * math.cos(angle)
            py = cy - ring * r_max * math.sin(angle)
            pts.append((px, py))
        pygame.draw.polygon(surf, DARK_GREY, pts, 1)

    # Draw axes
    for i in range(n):
        angle = math.pi/2 + 2*math.pi*i/n
        ex = cx + r_max * math.cos(angle)
        ey = cy - r_max * math.sin(angle)
        pygame.draw.line(surf, DARK_GREY, (cx, cy), (ex, ey), 1)
        lx = cx + (r_max+14) * math.cos(angle) - 16
        ly = cy - (r_max+14) * math.sin(angle) - 6
        surf.blit(font_sm.render(labels[i], True, GREY), (lx, ly))

    # Draw filled polygon
    pts = []
    for i, val in enumerate(values):
        angle = math.pi/2 + 2*math.pi*i/n
        px = cx + val * r_max * math.cos(angle)
        py = cy - val * r_max * math.sin(angle)
        pts.append((px, py))
    if len(pts) >= 3:
        s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        local_pts = [(p[0]-rect.x, p[1]-rect.y) for p in pts]
        pygame.draw.polygon(s, (*color, 100), local_pts)
        pygame.draw.polygon(s, (*color, 220), local_pts, 2)
        surf.blit(s, (rect.x, rect.y))


def draw_gauge(surf, rect, value, title="", label=""):
    """Draw a semicircle speedometer gauge."""
    draw_panel(surf, rect)
    font_sm = pygame.font.SysFont("monospace", 11)
    font_ti = pygame.font.SysFont("monospace", 13, bold=True)
    font_lg = pygame.font.SysFont("monospace", 20, bold=True)
    if title:
        surf.blit(font_ti.render(title, True, GREY), (rect.x+PANEL_PAD, rect.y+PANEL_PAD))

    cx = rect.centerx
    cy = rect.y + rect.height - 28
    r  = min(rect.width//2 - 20, rect.height - 50)

    # Draw arc segments (green → yellow → red)
    for deg in range(0, 180, 3):
        t     = deg / 180
        angle = math.pi - math.radians(deg)
        col   = (int(50+200*t), int(220-160*t), 60) if t < 0.5 else \
                (int(150+100*(t-0.5)*2), int(60+60*(1-(t-0.5)*2)), 60)
        x1 = cx + (r-8) * math.cos(angle)
        y1 = cy - (r-8) * math.sin(angle)
        x2 = cx + r      * math.cos(angle)
        y2 = cy - r      * math.sin(angle)
        pygame.draw.line(surf, col, (int(x1), int(y1)), (int(x2), int(y2)), 4)

    # Needle
    needle_angle = math.pi - math.pi * max(0, min(1, value))
    nx = cx + (r-4) * math.cos(needle_angle)
    ny = cy - (r-4) * math.sin(needle_angle)
    pygame.draw.line(surf, WHITE, (cx, cy), (int(nx), int(ny)), 3)
    pygame.draw.circle(surf, WHITE, (cx, cy), 5)

    # Value text
    val_col = GREEN if value < 0.4 else ORANGE if value < 0.7 else RED
    surf.blit(font_lg.render(f"{value:.3f}", True, val_col),
              (cx-28, cy-18))
    if label:
        surf.blit(font_sm.render(label, True, GREY), (cx-20, cy+6))


def draw_rover_map(surf, rect, cost_map, trajectory, plan_path,
                   rover_pos, goal, step, paused, title=""):
    draw_panel(surf, rect)
    font_sm = pygame.font.SysFont("monospace", 11)
    font_ti = pygame.font.SysFont("monospace", 13, bold=True)
    if title:
        surf.blit(font_ti.render(title, True, GREY), (rect.x+PANEL_PAD, rect.y+PANEL_PAD))

    H, W = cost_map.shape
    inner = pygame.Rect(rect.x+PANEL_PAD, rect.y+28,
                        rect.width-2*PANEL_PAD, rect.height-40)
    cell_w = inner.width  / W
    cell_h = inner.height / H

    # Draw cells
    max_cost = cost_map[cost_map < 999].max() if (cost_map < 999).any() else 1
    for i in range(H):
        for j in range(W):
            c    = cost_map[i, j]
            cx_  = int(inner.x + j * cell_w)
            cy_  = int(inner.y + i * cell_h)
            cw_  = max(1, int(cell_w))
            ch_  = max(1, int(cell_h))
            if c >= 999:
                col = (50, 20, 20)
            else:
                t   = c / max_cost
                col = (int(20+80*t), int(100+100*(1-t)), int(60+140*(1-t)))
            pygame.draw.rect(surf, col, (cx_, cy_, cw_, ch_))

    # Draw planned path
    if plan_path:
        for p in plan_path:
            px = int(inner.x + p[1] * cell_w + cell_w/2)
            py = int(inner.y + p[0] * cell_h + cell_h/2)
            pygame.draw.circle(surf, YELLOW, (px, py), 2)

    # Draw trajectory
    if len(trajectory) > 1:
        pts = [(int(inner.x + p[1]*cell_w + cell_w/2),
                int(inner.y + p[0]*cell_h + cell_h/2)) for p in trajectory]
        pygame.draw.lines(surf, CYAN, False, pts, 2)

    # Draw goal
    gx = int(inner.x + goal[1]*cell_w + cell_w/2)
    gy = int(inner.y + goal[0]*cell_h + cell_h/2)
    pygame.draw.circle(surf, RED, (gx, gy), int(cell_w*0.6))
    surf.blit(font_sm.render("G", True, WHITE), (gx-4, gy-5))

    # Draw rover as triangle (shows direction)
    rx = inner.x + rover_pos[1] * cell_w + cell_w/2
    ry = inner.y + rover_pos[0] * cell_h + cell_h/2
    if len(trajectory) >= 2:
        dy = trajectory[-1][0] - trajectory[-2][0]
        dx = trajectory[-1][1] - trajectory[-2][1]
        heading = math.atan2(dx, -dy) if (dx != 0 or dy != 0) else 0
    else:
        heading = 0
    ts = max(int(cell_w*1.1), 6)
    pts_t = [
        (rx + ts * math.sin(heading),         ry - ts * math.cos(heading)),
        (rx + ts * math.sin(heading+2.3),      ry - ts * math.cos(heading+2.3)),
        (rx + ts * math.sin(heading-2.3),      ry - ts * math.cos(heading-2.3)),
    ]
    pygame.draw.polygon(surf, LIME, [(int(p[0]), int(p[1])) for p in pts_t])

    # Step counter
    status = "PAUSED" if paused else f"Step {step}"
    surf.blit(font_sm.render(status, True, ORANGE if paused else CYAN),
              (rect.x + rect.width - 80, rect.y + PANEL_PAD))


def draw_status_bar(surf, rect, terrain_cls, h_score,
                    cam_conf, lidar_conf, step, total_steps):
    pygame.draw.rect(surf, (22, 22, 35), rect)
    pygame.draw.line(surf, PANEL_EDGE, (rect.x, rect.y), (rect.x+rect.width, rect.y), 1)
    font  = pygame.font.SysFont("monospace", 14, bold=True)
    font2 = pygame.font.SysFont("monospace", 13)

    # Agreement badge
    diff = abs(cam_conf - lidar_conf)
    if diff < 0.15:
        badge_col, badge_text = GREEN, "✓ SENSORS AGREE"
    elif diff < 0.35:
        badge_col, badge_text = ORANGE, "~ MINOR CONFLICT"
    else:
        badge_col, badge_text = RED, "✗ SENSOR CONFLICT"

    bw = 220
    pygame.draw.rect(surf, badge_col, (rect.x+10, rect.y+8, bw, 44), border_radius=6)
    surf.blit(font.render(badge_text, True, (10,10,10)),
              (rect.x+18, rect.y+12))
    surf.blit(font2.render(f"Δ={diff:.3f}", True, (10,10,10)),
              (rect.x+18, rect.y+30))

    # Terrain class
    tc_col = TERRAIN_COLORS.get(terrain_cls, WHITE)
    pygame.draw.rect(surf, tc_col, (rect.x+250, rect.y+8, 120, 44), border_radius=6)
    surf.blit(font.render(terrain_cls[:8], True, (20,20,20)),
              (rect.x+258, rect.y+22))

    # H-score
    hc = GREEN if h_score < 0.4 else ORANGE if h_score < 0.7 else RED
    surf.blit(font.render(f"H-score: {h_score:.3f}", True, hc),
              (rect.x+400, rect.y+12))

    # Step progress
    surf.blit(font2.render(f"Step {step}/{total_steps}", True, GREY),
              (rect.x+400, rect.y+32))

    # Controls
    ctrl = "SPACE=pause  R=reset  Q=quit"
    surf.blit(font2.render(ctrl, True, DARK_GREY),
              (rect.x + rect.width - 320, rect.y + 20))


# ─── Build a random cost map ───────────────────────────────────────────────────
def build_map(H=15, W=15, alpha=0.75, h_crit=0.7):
    classes    = TERRAIN_NAMES
    class_grid = [[random.choice(classes) for _ in range(W)] for _ in range(H)]
    cost_map   = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            cls  = class_grid[i][j]
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
    return cost_map, class_grid


# ─── Main simulation ───────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W_SCREEN, H_SCREEN))
    pygame.display.set_caption("ANS Mars Rover — Live Simulation")
    clock  = pygame.time.Clock()

    # Load real terrain images (with synthetic fallback)
    IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_images')
    TERRAIN_SURFS = load_terrain_images(IMG_DIR)

    # Load RL agent
    agent    = DoubleDQN()
    dqn_path = os.path.join(os.path.dirname(__file__), "..", "models", "dqn_rover.pt")
    if os.path.exists(dqn_path):
        agent.online_net.load_state_dict(torch.load(dqn_path, map_location="cpu"))
        print("Loaded trained RL agent")
    else:
        print("No trained agent found — using random policy")
    agent.epsilon = 0.05

    def reset_sim():
        cost_map, class_grid = build_map()
        H, W  = cost_map.shape
        start = (0, 0)
        goal  = (H-1, W-1)
        env   = MarsRoverEnv(cost_map, start, goal)
        env.reset()
        planner   = ThetaStar(cost_map)
        plan_path = planner.find_path(start, goal)
        state     = env.get_state()
        return env, cost_map, class_grid, plan_path, state, start, goal

    env, cost_map, class_grid, plan_path, state, start, goal = reset_sim()
    H, W = cost_map.shape

    trajectory    = [tuple(env.current_pos)]
    paused        = False
    done          = False
    step          = 0
    max_steps     = 300
    last_step_t   = time.time()

    # Current cell sensor readings
    cur_cls       = "bedrock"
    cur_geom      = get_geometry(cur_cls)
    cur_entropy   = 0.3
    cur_cam_conf  = 0.85
    cur_hscore    = 0.2
    cur_probs     = [0.05, 0.80, 0.10, 0.05]

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    env, cost_map, class_grid, plan_path, state, start, goal = reset_sim()
                    H, W       = cost_map.shape
                    trajectory = [tuple(env.current_pos)]
                    step, done = 0, False
                    last_step_t = time.time()

        # Advance rover
        if not paused and not done and (time.time() - last_step_t) >= STEP_DELAY:
            action            = agent.select_action(state)
            state, reward, done = env.step(action)
            step += 1
            trajectory.append(tuple(env.current_pos))
            last_step_t = time.time()

            # Update sensor readings for current cell
            ri  = int(np.clip(env.current_pos[0], 0, H-1))
            ci  = int(np.clip(env.current_pos[1], 0, W-1))
            cur_cls  = class_grid[ri][ci]
            cur_geom = get_geometry(cur_cls)

            # Simulate CNN probabilities with noise
            base_probs = {"soil":0.05,"bedrock":0.05,"sand":0.05,"big_rocks":0.05}
            base_probs[cur_cls] = 0.75
            raw = np.array([base_probs[c] + random.uniform(0,0.12) for c in TERRAIN_NAMES])
            raw /= raw.sum()
            cur_probs     = raw.tolist()
            cur_cam_conf  = max(cur_probs)
            cur_entropy   = -sum(p * math.log(p+1e-8) for p in cur_probs)
            cur_entropy   = min(cur_entropy / math.log(4), 1.0)
            u_fused       = compute_fusion(cur_entropy, cur_cam_conf,
                                           cur_geom["lidar_conf"], beta=0.5)
            cur_hscore    = compute_hscore(u_fused, cur_cls, alpha=0.75)

            if done and step < max_steps:
                paused = True  # pause at end so user can see result

        # ── DRAW ──────────────────────────────────────────────────────────────
        screen.fill(BG)

        # Row separators
        sep_y = H_ROW
        pygame.draw.line(screen, PANEL_EDGE, (0, sep_y), (W_SCREEN, sep_y), 1)
        pygame.draw.line(screen, PANEL_EDGE, (W_LEFT, 0), (W_LEFT, H_SCREEN-H_STATUS), 1)
        pygame.draw.line(screen, PANEL_EDGE, (W_LEFT+W_MID, 0), (W_LEFT+W_MID, H_SCREEN-H_STATUS), 1)

        # Row labels
        font_row = pygame.font.SysFont("monospace", 12, bold=True)
        screen.blit(font_row.render("ROW 1 — SIMULATION VIEW", True, DARK_GREY), (8, 4))
        screen.blit(font_row.render("ROW 2 — SENSOR VIEW", True, DARK_GREY), (8, H_ROW+4))

        pad = PANEL_PAD

        # ── ROW 1 ─────────────────────────────────────────────────────────────
        # Left: rover map
        r1_left = pygame.Rect(pad, 18, W_LEFT-2*pad, H_ROW-22)
        draw_rover_map(screen, r1_left, cost_map, trajectory, plan_path,
                       env.current_pos, goal, step, paused,
                       title="Rover Navigation Map")

        # Middle: CNN terrain prediction
        r1_mid = pygame.Rect(W_LEFT+pad, 18, W_MID-2*pad, H_ROW-22)
        bar_cols = [TERRAIN_COLORS[c] for c in TERRAIN_NAMES]
        draw_bar_chart(screen, r1_mid, cur_probs, TERRAIN_NAMES, bar_cols,
                       title="CNN Terrain Prediction")

        # Right: LiDAR geometry reading
        r1_right = pygame.Rect(W_LEFT+W_MID+pad, 18, W_RIGHT-2*pad, H_ROW-22)
        lidar_vals = [
            cur_geom["slope"],
            cur_geom["roughness"],
            cur_geom["lidar_conf"],
            min(cur_hscore, 1.0),
        ]
        lidar_cols = [CYAN, PURPLE, GREEN, RED]
        lidar_lbls = ["slope", "rough", "lidar", "H-sc"]
        draw_bar_chart(screen, r1_right, lidar_vals, lidar_lbls, lidar_cols,
                       title="LiDAR Geometry Reading")

        # ── ROW 2 ─────────────────────────────────────────────────────────────
        row2_y = H_ROW + 18

        # Left: actual terrain image (fills panel minus title bar)
        r2_left = pygame.Rect(pad, row2_y, W_LEFT-2*pad, H_ROW-22)
        draw_panel(screen, r2_left, title="Terrain Image (Current Cell)")
        title_h = 28
        img_rect = pygame.Rect(r2_left.x+2, r2_left.y+title_h,
                               r2_left.width-4, r2_left.height-title_h-2)
        t_surf   = pygame.transform.scale(TERRAIN_SURFS[cur_cls],
                                          (img_rect.width, img_rect.height))
        screen.blit(t_surf, (img_rect.x, img_rect.y))
        font_cls = pygame.font.SysFont("monospace", 15, bold=True)
        tc_col   = TERRAIN_COLORS[cur_cls]
        pygame.draw.rect(screen, tc_col,
                         (r2_left.x+10, r2_left.y+r2_left.height-28, 140, 20),
                         border_radius=4)
        screen.blit(font_cls.render(cur_cls, True, (20,20,20)),
                    (r2_left.x+16, r2_left.y+r2_left.height-26))

        # Middle: Camera — confidence ring + entropy gauge
        r2_mid = pygame.Rect(W_LEFT+pad, row2_y, W_MID-2*pad, H_ROW-22)
        draw_panel(screen, r2_mid, title="Camera Sensor")

        # Confidence donut chart
        cx_d = r2_mid.x + r2_mid.width // 2
        cy_d = r2_mid.y + 85
        r_out, r_in = 55, 32
        start_angle = -math.pi / 2
        for i, (prob, cls) in enumerate(zip(cur_probs, TERRAIN_NAMES)):
            end_angle = start_angle + 2 * math.pi * prob
            col = TERRAIN_COLORS[cls]
            for deg in range(int(math.degrees(start_angle)),
                             int(math.degrees(end_angle)), 2):
                rad = math.radians(deg)
                for r_val in range(r_in, r_out):
                    px = cx_d + int(r_val * math.cos(rad))
                    py = cy_d + int(r_val * math.sin(rad))
                    screen.set_at((px, py), col)
            start_angle = end_angle

        # Small terrain thumbnail with entropy overlay (top-right of Camera panel)
        thumb_size  = 80
        thumb_x     = r2_mid.x + r2_mid.width - thumb_size - 10
        thumb_y     = r2_mid.y + 28
        thumb_surf  = pygame.transform.scale(TERRAIN_SURFS[cur_cls],
                                             (thumb_size, thumb_size))
        screen.blit(thumb_surf, (thumb_x, thumb_y))
        overlay = pygame.Surface((thumb_size, thumb_size), pygame.SRCALPHA)
        overlay.fill((255, 0, 0, int(cur_entropy * 180)))
        screen.blit(overlay, (thumb_x, thumb_y))
        pygame.draw.rect(screen, DARK_GREY,
                         (thumb_x, thumb_y, thumb_size, thumb_size), 1)

        # Entropy indicator
        ent_col = GREEN if cur_entropy < 0.3 else ORANGE if cur_entropy < 0.65 else RED
        font_e  = pygame.font.SysFont("monospace", 13, bold=True)
        font_e2 = pygame.font.SysFont("monospace", 11)
        screen.blit(font_e.render(f"Entropy: {cur_entropy:.3f}", True, ent_col),
                    (r2_mid.x+pad, r2_mid.y+150))
        screen.blit(font_e2.render("LOW=confident  HIGH=uncertain", True, DARK_GREY),
                    (r2_mid.x+pad, r2_mid.y+166))

        # Cam conf bar
        bw = r2_mid.width - 20
        bh = 14
        by = r2_mid.y + 190
        pygame.draw.rect(screen, DARK_GREY, (r2_mid.x+10, by, bw, bh), border_radius=4)
        pygame.draw.rect(screen, GREEN,
                         (r2_mid.x+10, by, int(bw*cur_cam_conf), bh), border_radius=4)
        screen.blit(font_e2.render(f"Cam conf: {cur_cam_conf:.2f}", True, WHITE),
                    (r2_mid.x+10, by+16))

        # Right: LiDAR — radar chart + H-score speedometer
        r2_right = pygame.Rect(W_LEFT+W_MID+pad, row2_y, W_RIGHT-2*pad, H_ROW-22)
        r2_radar = pygame.Rect(r2_right.x, r2_right.y,
                               r2_right.width, r2_right.height//2 + 20)
        r2_gauge = pygame.Rect(r2_right.x, r2_right.y + r2_right.height//2 + 20,
                               r2_right.width, r2_right.height//2 - 20)

        radar_color = (40, 180, 220) if cur_hscore < 0.4 else \
                      (220, 160, 40) if cur_hscore < 0.7 else (220, 60, 60)

        draw_radar(screen, r2_radar,
                   [cur_geom["slope"], cur_geom["roughness"],
                    cur_geom["lidar_conf"], min(cur_hscore, 1.0)],
                   ["slope", "rough", "lidar", "H"],
                   radar_color, title="LiDAR Radar")

        draw_gauge(screen, r2_gauge, min(cur_hscore, 1.0),
                   title="H-score Gauge", label="danger")

        # ── Status bar ────────────────────────────────────────────────────────
        status_rect = pygame.Rect(0, H_SCREEN-H_STATUS, W_SCREEN, H_STATUS)
        draw_status_bar(screen, status_rect, cur_cls, cur_hscore,
                        cur_cam_conf, cur_geom["lidar_conf"],
                        step, max_steps)

        # Mission complete overlay
        if done:
            font_big = pygame.font.SysFont("monospace", 36, bold=True)
            msg      = "MISSION COMPLETE!" if step < max_steps else "MAX STEPS REACHED"
            col_msg  = GREEN if step < max_steps else ORANGE
            txt      = font_big.render(msg, True, col_msg)
            screen.blit(txt, (W_SCREEN//2 - txt.get_width()//2, H_SCREEN//2 - 20))
            sub = pygame.font.SysFont("monospace", 16).render(
                "Press R to run again", True, WHITE)
            screen.blit(sub, (W_SCREEN//2 - sub.get_width()//2, H_SCREEN//2 + 30))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
