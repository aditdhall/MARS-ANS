"""
ANS Mars Rover — Live Simulation Demo  (v2 — polished)
========================================================
Controls:
  SPACE  — pause / resume
  R      — reset with new random map
  Q/ESC  — quit
"""

import math, os, random, sys, time
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(__file__))
import config
from perception import get_geometry, compute_fusion, compute_hscore
from planner import ThetaStar, DStarLite
from rl_agent import MarsRoverEnv, DoubleDQN
import torch

# ── Screen layout ──────────────────────────────────────────────────────────────
W, H        = 1400, 820
H_ROW       = 360
H_STATUS    = 56
W_LEFT      = 500
W_MID       = 380
W_RIGHT     = W - W_LEFT - W_MID
PAD         = 12
FPS         = 60
STEP_DELAY  = 0.75

# ── Colour palette ─────────────────────────────────────────────────────────────
BG          = (10, 12, 22)
PANEL       = (18, 22, 38)
PANEL2      = (22, 28, 48)
BORDER      = (45, 55, 90)
GLOW_BLUE   = (40, 120, 220)
GLOW_CYAN   = (40, 210, 220)
GLOW_GREEN  = (40, 210, 110)
GLOW_RED    = (220, 55, 55)
GLOW_ORANGE = (240, 155, 30)
GLOW_PURPLE = (160, 70, 220)
WHITE       = (240, 245, 255)
GREY        = (130, 140, 165)
DARK_GREY   = (55, 62, 82)
LIME        = (100, 240, 80)
YELLOW      = (240, 220, 50)

TERRAIN_COL = {
    "soil":      (170, 130,  85),
    "bedrock":   (105, 108, 128),
    "sand":      (205, 188, 130),
    "big_rocks": (118,  72,  48),
}
TERRAIN_GLOW = {
    "soil":      (200, 160, 100),
    "bedrock":   (130, 135, 160),
    "sand":      (230, 210, 150),
    "big_rocks": (160,  95,  60),
}
NAMES = ["soil", "bedrock", "sand", "big_rocks"]

# ── Font cache ─────────────────────────────────────────────────────────────────
# SysFont("monospace") on macOS falls back to a bitmap font that renders as
# solid blocks at small bold sizes. Resolve explicit faces via match_font and
# fall back to pygame's bundled FreeSans (Font(None, ...)) if none are present.
_fonts = {}
_MONO_STACK  = ["menlo", "sfmono", "monaco", "consolas", "dejavusansmono",
                "couriernew", "liberationmono"]
_TITLE_STACK = ["sfprotext", "helveticaneue", "helvetica", "arial",
                "segoeui", "dejavusans", "liberationsans"]

def _resolve(stack, size, bold):
    for name in stack:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            f = pygame.font.Font(path, size)
            if bold:
                f.set_bold(True)
            return f
    f = pygame.font.Font(None, size)
    if bold:
        f.set_bold(True)
    return f

def font(size, bold=False):
    key = ("mono", size, bold)
    if key not in _fonts:
        _fonts[key] = _resolve(_MONO_STACK, size, bold)
    return _fonts[key]

def title_font(size, bold=True):
    key = ("title", size, bold)
    if key not in _fonts:
        _fonts[key] = _resolve(_TITLE_STACK, size, bold)
    return _fonts[key]

# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_rounded(surf, col, rect, r=10, border=None, border_col=None, width=2):
    pygame.draw.rect(surf, col, rect, border_radius=r)
    if border:
        pygame.draw.rect(surf, border_col or BORDER, rect, width, border_radius=r)

def glow_line(surf, col, p1, p2, w=2, glow=6):
    gc = tuple(min(255, c + 80) for c in col)
    for ww in range(glow, 0, -2):
        alpha_col = tuple(int(c * ww/glow) for c in gc)
        pygame.draw.line(surf, alpha_col, p1, p2, ww)
    pygame.draw.line(surf, col, p1, p2, w)

def glow_circle(surf, col, pos, r, w=2, glow=8):
    for ww in range(glow, 0, -2):
        gc = tuple(int(c * ww/glow) for c in col)
        pygame.draw.circle(surf, gc, pos, r + ww//2, ww)
    pygame.draw.circle(surf, col, pos, r, w)

def draw_panel(surf, rect, title=None, accent=GLOW_BLUE, r=12):
    # Subtle vertical gradient fill for depth
    grad = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    for i in range(rect.height):
        t = i / max(rect.height-1, 1)
        col = (
            int(PANEL[0] * (1 - 0.25*t) + PANEL2[0] * 0.25*t),
            int(PANEL[1] * (1 - 0.25*t) + PANEL2[1] * 0.25*t),
            int(PANEL[2] * (1 - 0.25*t) + PANEL2[2] * 0.25*t),
            255,
        )
        pygame.draw.line(grad, col, (0, i), (rect.width, i))
    mask = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=r)
    grad.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    surf.blit(grad, rect.topleft)

    # Top accent bar (kept short of the title so it doesn't clip letters)
    top = pygame.Rect(rect.x + 10, rect.y + 3, rect.width - 20, 2)
    pygame.draw.rect(surf, accent, top, border_radius=2)

    # Border
    pygame.draw.rect(surf, BORDER, rect, 1, border_radius=r)

    if title:
        t = title.strip()
        label  = title_font(15).render(t, True, WHITE)
        shadow = title_font(15).render(t, True, (0, 0, 0))
        ty = rect.y + 8
        surf.blit(shadow, (rect.x + PAD + 1, ty + 1))
        surf.blit(label,  (rect.x + PAD,     ty))
        # Thin accent underline under the title (sits inside the 30px header band)
        uw = min(label.get_width() + 8, rect.width - 2*PAD)
        uy = ty + label.get_height() + 1
        pygame.draw.line(surf, accent,
                         (rect.x + PAD, uy),
                         (rect.x + PAD + uw, uy), 2)

def pct_bar(surf, rect, val, col, bg=DARK_GREY, r=4):
    draw_rounded(surf, bg, rect, r)
    filled = pygame.Rect(rect.x, rect.y, int(rect.width * max(0, min(1, val))), rect.height)
    if filled.width > 0:
        draw_rounded(surf, col, filled, r)

def load_images(img_dir):
    surfs = {}
    for cls in NAMES:
        path = os.path.join(img_dir, f"{cls}.JPG")
        try:
            surfs[cls] = pygame.image.load(path).convert()
            print(f"  Loaded {cls}")
        except Exception:
            s = pygame.Surface((200, 200))
            s.fill(TERRAIN_COL[cls])
            surfs[cls] = s
    return surfs

# ── Build cost map ─────────────────────────────────────────────────────────────
def build_map(H=15, W=15, alpha=0.75, h_crit=0.7, start=None, goal=None):
    grid     = [[random.choice(NAMES) for _ in range(W)] for _ in range(H)]
    cost_map = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            cls  = grid[i][j]
            geom = get_geometry(cls)
            h    = compute_hscore(geom["lidar_conf"] * 0.3, cls, alpha)
            cost_map[i, j] = 999 if h > h_crit else (
                0.3*config.TERRAIN_COST[cls]
                + 0.2*geom["slope"]
                + 0.2*geom["roughness"]
                + 0.3*h
            )

    if start is None:
        # Pick random start/goal on passable cells with >=10 cell separation
        passable = [(i, j) for i in range(H) for j in range(W)
                    if cost_map[i, j] < 999]
        for _ in range(500):
            s = random.choice(passable)
            g = random.choice(passable)
            if math.hypot(s[0]-g[0], s[1]-g[1]) >= 10:
                start, goal = s, g
                break
        if start is None:
            start, goal = (0, 0), (H-1, W-1)

    cost_map[start] = 0.15
    cost_map[goal]  = 0.15
    return cost_map, grid, start, goal

# ── Rover map panel ────────────────────────────────────────────────────────────
def draw_map_panel(surf, rect, cost_map, class_grid, traj, theta_path,
                   dstar_path, rover_pos, goal, step, paused, conflict,
                   replan_cells=None):
    draw_panel(surf, rect, "  Rover Navigation Map",
               accent=GLOW_RED if conflict else GLOW_CYAN)
    GH, GW = cost_map.shape
    inner = pygame.Rect(rect.x+PAD, rect.y+30,
                        rect.width-2*PAD, rect.height-42)
    cw = inner.width / GW
    ch = inner.height / GH

    max_c = cost_map[cost_map < 999].max() if (cost_map < 999).any() else 1

    for i in range(GH):
        for j in range(GW):
            c  = cost_map[i, j]
            x_ = int(inner.x + j*cw)
            y_ = int(inner.y + i*ch)
            w_ = max(1, int(cw))
            h_ = max(1, int(ch))
            if c >= 999:
                col = (35, 15, 15)
                pygame.draw.rect(surf, col, (x_, y_, w_, h_))
                pygame.draw.rect(surf, (55, 25, 25), (x_, y_, w_, h_), 1)
            else:
                base = TERRAIN_COL.get(class_grid[i][j], (80, 80, 80))
                t    = c / max_c
                col  = tuple(int(b * (1 - 0.3*t)) for b in base)
                pygame.draw.rect(surf, col, (x_, y_, w_, h_))

    # Theta* path (white dots)
    for p in theta_path:
        px = int(inner.x + p[1]*cw + cw/2)
        py = int(inner.y + p[0]*ch + ch/2)
        pygame.draw.circle(surf, (180, 180, 180), (px, py), 2)

    # D* Lite replan path (glowing orange)
    if dstar_path and len(dstar_path) > 1:
        pts = [(int(inner.x + p[1]*cw + cw/2),
                int(inner.y + p[0]*ch + ch/2)) for p in dstar_path]
        for i in range(len(pts)-1):
            glow_line(surf, GLOW_ORANGE, pts[i], pts[i+1], w=2, glow=5)

    # Trajectory
    if len(traj) > 1:
        pts = [(int(inner.x + p[1]*cw + cw/2),
                int(inner.y + p[0]*ch + ch/2)) for p in traj]
        for i in range(len(pts)-1):
            alpha = int(80 + 175 * i / len(pts))
            col   = (40, alpha, alpha)
            pygame.draw.line(surf, col, pts[i], pts[i+1], 2)

    # D* replan markers — small red X at each cell where D* was triggered
    if replan_cells:
        for (rr, cc) in replan_cells:
            if 0 <= rr < GH and 0 <= cc < GW:
                mx = int(inner.x + cc*cw + cw/2)
                my = int(inner.y + rr*ch + ch/2)
                d  = max(int(min(cw, ch) * 0.35), 4)
                pygame.draw.line(surf, GLOW_RED, (mx-d, my-d), (mx+d, my+d), 2)
                pygame.draw.line(surf, GLOW_RED, (mx-d, my+d), (mx+d, my-d), 2)

    # Goal beacon (pulsing)
    gx = int(inner.x + goal[1]*cw + cw/2)
    gy = int(inner.y + goal[0]*ch + ch/2)
    pulse = int(4 + 3 * math.sin(time.time() * 4))
    glow_circle(surf, GLOW_RED, (gx, gy), pulse, w=2, glow=8)
    surf.blit(font(10, True).render("G", True, WHITE), (gx-4, gy-5))

    # Rover triangle
    rx = inner.x + rover_pos[1]*cw + cw/2
    ry = inner.y + rover_pos[0]*ch + ch/2
    if len(traj) >= 2:
        dy = traj[-1][0] - traj[-2][0]
        dx = traj[-1][1] - traj[-2][1]
        heading = math.atan2(dx, -dy) if (dx or dy) else 0
    else:
        heading = 0
    ts = max(int(cw*1.3), 7)
    tri = [
        (rx + ts*math.sin(heading),      ry - ts*math.cos(heading)),
        (rx + ts*math.sin(heading+2.3),  ry - ts*math.cos(heading+2.3)),
        (rx + ts*math.sin(heading-2.3),  ry - ts*math.cos(heading-2.3)),
    ]
    # Glow
    glow_col = GLOW_RED if conflict else LIME
    pygame.draw.polygon(surf, tuple(c//3 for c in glow_col),
                        [(int(p[0]), int(p[1])) for p in tri])
    pygame.draw.polygon(surf, glow_col,
                        [(int(p[0]), int(p[1])) for p in tri])

    # Legend
    lx, ly = rect.x + PAD, rect.y + rect.height - 26
    pygame.draw.line(surf, (180,180,180), (lx, ly+4), (lx+18, ly+4), 1)
    surf.blit(font(10).render("Theta*", True, GREY), (lx+22, ly))
    lx2 = lx + 80
    glow_line(surf, GLOW_ORANGE, (lx2, ly+4), (lx2+18, ly+4), w=2, glow=3)
    surf.blit(font(10).render("D* replan", True, GREY), (lx2+22, ly))

    # Status
    st = "PAUSED" if paused else f"Step {step}/300"
    surf.blit(font(11, True).render(st, True, GLOW_ORANGE if paused else GLOW_CYAN),
              (rect.x + rect.width - 90, rect.y + PAD + 2))

# ── CNN bar chart ──────────────────────────────────────────────────────────────
def draw_cnn_panel(surf, rect, probs, cam_conf, entropy):
    draw_panel(surf, rect, "  CNN Terrain Prediction", accent=GLOW_BLUE)
    inner = rect.inflate(-2*PAD, -8)
    inner.y += 30
    inner.height -= 30

    bh = 22
    gap = 10
    total_h = len(NAMES) * (bh + gap)
    start_y = inner.y + (inner.height - total_h) // 2

    for i, (cls, prob) in enumerate(zip(NAMES, probs)):
        y = start_y + i*(bh+gap)
        col = TERRAIN_COL[cls]
        gc  = TERRAIN_GLOW[cls]

        # Background track
        draw_rounded(surf, DARK_GREY,
                     pygame.Rect(inner.x+70, y, inner.width-70, bh), 5)
        # Fill
        fw = int((inner.width-70) * prob)
        if fw > 0:
            draw_rounded(surf, col,
                         pygame.Rect(inner.x+70, y, fw, bh), 5)
            # Glow edge
            pygame.draw.rect(surf, gc,
                             (inner.x+70+fw-3, y+2, 3, bh-4),
                             border_radius=3)

        # Label
        surf.blit(font(11, True).render(cls[:8], True, WHITE),
                  (inner.x, y + 4))
        # Value
        surf.blit(font(11, True).render(f"{prob:.2f}", True, WHITE),
                  (inner.x + inner.width - 35, y + 4))

    # Confidence arc at bottom
    cy = inner.y + inner.height - 18
    surf.blit(font(11).render(f"Conf: {cam_conf:.2f}", True, GLOW_GREEN), (inner.x, cy))
    pct_bar(surf, pygame.Rect(inner.x+80, cy+3, inner.width-80, 10),
            cam_conf, GLOW_GREEN)

# ── LiDAR bar panel ────────────────────────────────────────────────────────────
def draw_lidar_panel(surf, rect, geom, hscore):
    draw_panel(surf, rect, "  LiDAR Geometry", accent=GLOW_PURPLE)
    inner = rect.inflate(-2*PAD, -8)
    inner.y += 30

    vals  = [geom["slope"], geom["roughness"], geom["lidar_conf"], min(hscore, 1.0)]
    lbls  = ["Slope", "Roughness", "LiDAR Conf", "H-Score"]
    cols  = [GLOW_CYAN, GLOW_PURPLE, GLOW_GREEN, GLOW_RED]

    bh = 20
    gap = 14
    for i, (v, lbl, col) in enumerate(zip(vals, lbls, cols)):
        y = inner.y + i*(bh+gap)
        surf.blit(font(11, True).render(lbl, True, WHITE), (inner.x, y))
        bar_rect = pygame.Rect(inner.x+90, y+2, inner.width-130, bh-4)
        pct_bar(surf, bar_rect, v, col)
        surf.blit(font(11, True).render(f"{v:.2f}", True, col),
                  (inner.x + inner.width - 35, y+2))

# ── Terrain image panel ────────────────────────────────────────────────────────
def draw_terrain_panel(surf, rect, img_surf, cls, entropy, title="  Terrain Image"):
    tc = TERRAIN_COL.get(cls, WHITE)
    draw_panel(surf, rect, title, accent=tc)
    ih = rect.height - 38
    t_surf = pygame.transform.scale(img_surf, (rect.width-4, ih))
    surf.blit(t_surf, (rect.x+2, rect.y+30))

    # Entropy overlay (red tint when uncertain)
    ov = pygame.Surface((rect.width-4, ih), pygame.SRCALPHA)
    ov.fill((220, 30, 30, int(entropy * 30)))
    surf.blit(ov, (rect.x+2, rect.y+30))

    # Class badge
    badge = pygame.Rect(rect.x+PAD, rect.y+rect.height-30, 130, 22)
    draw_rounded(surf, tc, badge, 6)
    surf.blit(font(12, True).render(cls, True, (20,20,20)),
              (badge.x+8, badge.y+4))

# ── Camera sensor panel ────────────────────────────────────────────────────────
def draw_camera_panel(surf, rect, probs, entropy, cam_conf, img_surf):
    draw_panel(surf, rect, "  Camera Sensor", accent=GLOW_BLUE)

    # Donut chart
    cx = rect.x + rect.width//2 - 30
    cy = rect.y + 110
    ro, ri = 60, 36
    ang = -math.pi/2
    for prob, cls in zip(probs, NAMES):
        end = ang + 2*math.pi*prob
        col = TERRAIN_COL[cls]
        for deg in range(int(math.degrees(ang)), int(math.degrees(end)), 2):
            rad = math.radians(deg)
            for rv in range(ri, ro):
                px = cx + int(rv*math.cos(rad))
                py = cy + int(rv*math.sin(rad))
                if 0 <= px < W and 0 <= py < H:
                    surf.set_at((px, py), col)
        ang = end

    # Inner circle
    pygame.draw.circle(surf, PANEL, (cx, cy), ri)
    dominant = NAMES[int(np.argmax(probs))]
    dc = TERRAIN_COL[dominant]
    surf.blit(font(10, True).render(dominant[:4], True, dc), (cx-16, cy-6))

    # Thumbnail with entropy overlay (top right)
    th = 80
    tx = rect.x + rect.width - th - PAD
    ty = rect.y + 34
    thumb = pygame.transform.scale(img_surf, (th, th))
    surf.blit(thumb, (tx, ty))
    ov = pygame.Surface((th, th), pygame.SRCALPHA)
    ov.fill((255, 40, 40, int(entropy * 200)))
    surf.blit(ov, (tx, ty))
    pygame.draw.rect(surf, GLOW_RED if entropy > 0.5 else GLOW_GREEN,
                     (tx, ty, th, th), 2)
    surf.blit(font(9).render("uncertainty", True, GREY), (tx, ty+th+2))

    # Entropy bar
    ey = rect.y + 190
    ec = GLOW_GREEN if entropy < 0.3 else GLOW_ORANGE if entropy < 0.65 else GLOW_RED
    surf.blit(font(12, True).render(f"Entropy: {entropy:.3f}", True, ec),
              (rect.x+PAD, ey))
    pct_bar(surf, pygame.Rect(rect.x+PAD, ey+20, rect.width-2*PAD, 10),
            entropy/1.4, ec)

    # Confidence
    surf.blit(font(11).render(f"Cam conf: {cam_conf:.2f}", True, GLOW_GREEN),
              (rect.x+PAD, ey+38))
    pct_bar(surf, pygame.Rect(rect.x+PAD, ey+54, rect.width-2*PAD, 8),
            cam_conf, GLOW_GREEN)

# ── LiDAR radar + gauge ────────────────────────────────────────────────────────
def draw_lidar_sensor_panel(surf, rect, geom, hscore):
    draw_panel(surf, rect, "  LiDAR Analysis", accent=GLOW_PURPLE)

    # Radar chart
    cx = rect.centerx
    cy = rect.y + 130
    rm = 70
    vals  = [geom["slope"], geom["roughness"], geom["lidar_conf"], min(hscore,1)]
    lbls  = ["slope", "rough", "lidar", "H"]
    radar_col = (40,180,220) if hscore < 0.4 else \
                (220,160,40) if hscore < 0.7 else (220,60,60)

    # Grid rings
    for ring in [0.33, 0.66, 1.0]:
        pts = []
        for i in range(4):
            a  = math.pi/2 + 2*math.pi*i/4
            pts.append((int(cx+ring*rm*math.cos(a)), int(cy-ring*rm*math.sin(a))))
        pygame.draw.polygon(surf, DARK_GREY, pts, 1)

    # Axes + labels
    for i in range(4):
        a  = math.pi/2 + 2*math.pi*i/4
        ex = cx + rm*math.cos(a)
        ey = cy - rm*math.sin(a)
        pygame.draw.line(surf, DARK_GREY, (cx,cy), (int(ex),int(ey)), 1)
        lx = cx + (rm+16)*math.cos(a) - 14
        ly = cy - (rm+16)*math.sin(a) - 6
        surf.blit(font(10).render(lbls[i], True, GREY), (int(lx), int(ly)))

    # Filled polygon
    pts = []
    for i, v in enumerate(vals):
        a  = math.pi/2 + 2*math.pi*i/4
        pts.append((int(cx+v*rm*math.cos(a)), int(cy-v*rm*math.sin(a))))
    if len(pts) >= 3:
        s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        lp = [(p[0]-rect.x, p[1]-rect.y) for p in pts]
        pygame.draw.polygon(s, (*radar_col, 90), lp)
        pygame.draw.polygon(s, (*radar_col, 200), lp, 2)
        surf.blit(s, (rect.x, rect.y))

    # H-score gauge (mini speedometer)
    gx, gy = cx, rect.y + rect.height - 55
    gr = 40
    for deg in range(0, 180, 3):
        t = deg/180
        a = math.pi - math.radians(deg)
        c = (int(50+200*t), int(220-160*t), 60) if t < 0.5 else \
            (int(150+100*(t-.5)*2), int(60+60*(1-(t-.5)*2)), 60)
        x1 = gx+(gr-6)*math.cos(a); y1 = gy-(gr-6)*math.sin(a)
        x2 = gx+gr*math.cos(a);     y2 = gy-gr*math.sin(a)
        pygame.draw.line(surf, c, (int(x1),int(y1)), (int(x2),int(y2)), 3)

    na = math.pi - math.pi*max(0,min(1,hscore))
    nx = gx+(gr-3)*math.cos(na); ny = gy-(gr-3)*math.sin(na)
    pygame.draw.line(surf, WHITE, (gx,gy), (int(nx),int(ny)), 2)
    pygame.draw.circle(surf, WHITE, (gx,gy), 4)

    hc = GLOW_GREEN if hscore < 0.4 else GLOW_ORANGE if hscore < 0.7 else GLOW_RED
    surf.blit(font(14, True).render(f"{hscore:.3f}", True, hc), (gx-22, gy-12))
    surf.blit(font(9).render("H-score", True, GREY), (gx-18, gy+6))

# ── Status bar ─────────────────────────────────────────────────────────────────
def draw_status(surf, rect, cls, hscore, cam_conf, lidar_conf, step,
                conflict, dstar_triggered, tick, replan_count=0):
    pygame.draw.rect(surf, (12, 14, 26), rect)
    pygame.draw.line(surf, BORDER, rect.topleft,
                     (rect.x+rect.width, rect.y), 1)

    diff = abs(cam_conf - lidar_conf)
    if diff < 0.15:
        bc, bt = GLOW_GREEN, "✓  SENSORS AGREE"
    elif diff < 0.35:
        bc, bt = GLOW_ORANGE, "~  MINOR CONFLICT"
    else:
        bc, bt = GLOW_RED, "✗  SENSOR CONFLICT"

    # Pulse effect on conflict
    if conflict:
        pulse = int(200 + 55 * math.sin(tick * 0.3))
        bc = (pulse, 50, 50)

    bw = 240
    draw_rounded(surf, bc, pygame.Rect(rect.x+10, rect.y+8, bw, 40), 8)
    surf.blit(font(13, True).render(bt, True, (10,10,10)),
              (rect.x+20, rect.y+14))
    surf.blit(font(10).render(f"Δ={diff:.3f}", True, (10,10,10)),
              (rect.x+20, rect.y+30))

    # Terrain badge
    tc = TERRAIN_COL.get(cls, WHITE)
    draw_rounded(surf, tc, pygame.Rect(rect.x+270, rect.y+8, 120, 40), 8)
    surf.blit(font(12, True).render(cls, True, (20,20,20)),
              (rect.x+278, rect.y+22))

    # H-score
    hc = GLOW_GREEN if hscore < 0.4 else GLOW_ORANGE if hscore < 0.7 else GLOW_RED
    surf.blit(font(14, True).render(f"H: {hscore:.3f}", True, hc),
              (rect.x+415, rect.y+10))

    # Progress bar
    pbar = pygame.Rect(rect.x+415, rect.y+32, 160, 10)
    pct_bar(surf, pbar, step/300, GLOW_CYAN)
    surf.blit(font(10).render(f"Step {step}/300", True, GREY),
              (rect.x+580, rect.y+29))

    # D* indicator
    if dstar_triggered:
        surf.blit(font(12, True).render("D* REPLAN", True, GLOW_ORANGE),
                  (rect.x+680, rect.y+18))

    # D* replan counter
    surf.blit(font(11, True).render(f"D* replans: {replan_count}", True, GLOW_ORANGE),
              (rect.x+800, rect.y+20))

    # Controls
    surf.blit(font(11).render("SPACE=pause  R=reset  Q=quit", True, DARK_GREY),
              (rect.x + rect.width - 290, rect.y + 20))

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("ANS Mars Rover — Live Simulation")
    clock  = pygame.time.Clock()

    IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_images')
    IMGS    = load_images(IMG_DIR)

    agent    = DoubleDQN()
    dqn_path = os.path.join(os.path.dirname(__file__), "..", "models", "dqn_rover.pt")
    if os.path.exists(dqn_path):
        agent.online_net.load_state_dict(torch.load(dqn_path, map_location="cpu"))
        print("Loaded trained RL agent")
    agent.epsilon = 0.05

    def reset_sim():
        cm, cg, s, g = build_map()
        env    = MarsRoverEnv(cm, s, g)
        env.reset()
        ts     = ThetaStar(cm)
        tp     = ts.find_path(s, g)
        dl     = DStarLite(cm.copy())
        dl.find_path(s, g)
        return env, cm, cg, tp, dl, [], env.get_state(), s, g

    env, cost_map, class_grid, theta_path, dl, dstar_path, state, start, goal = reset_sim()
    GH, GW = cost_map.shape

    traj          = [tuple(env.current_pos)]
    paused        = False
    done          = False
    step          = 0
    last_t        = time.time()
    tick          = 0

    cur_cls       = "bedrock"
    cur_geom      = get_geometry(cur_cls)
    cur_entropy   = 0.3
    cur_cam_conf  = 0.85
    cur_hscore    = 0.2
    cur_probs     = [0.05, 0.80, 0.10, 0.05]
    preview_cls   = "bedrock"
    conflict      = False
    dstar_active  = False
    replan_count  = 0
    replan_cells  = []

    running = True
    while running:
        clock.tick(FPS)
        tick += 1

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    env, cost_map, class_grid, theta_path, dl, dstar_path, \
                        state, start, goal = reset_sim()
                    GH, GW = cost_map.shape
                    traj   = [tuple(env.current_pos)]
                    step = 0; done = False; dstar_active = False; conflict = False
                    replan_count = 0; replan_cells = []
                    last_t = time.time()

        # ── Step rover ────────────────────────────────────────────────────────
        if not paused and not done and (time.time() - last_t) >= STEP_DELAY:
            if dstar_path and len(dstar_path) > 1:
                cur_r, cur_c = env.current_pos
                goal_r, goal_c = goal
                # Find next waypoint closer to goal than current position
                cur_dist_to_goal = math.hypot(cur_r - goal_r, cur_c - goal_c)
                next_wp = None
                for wp in dstar_path:
                    wp_dist = math.hypot(wp[0] - goal_r, wp[1] - goal_c)
                    rover_to_wp = math.hypot(wp[0] - cur_r, wp[1] - cur_c)
                    if rover_to_wp > 0.5 and wp_dist < cur_dist_to_goal:
                        next_wp = wp
                        break
                if next_wp:
                    dr = next_wp[0] - cur_r
                    dc = next_wp[1] - cur_c
                    angle = math.atan2(dc, -dr)
                    action = int(round(angle / (math.pi/4))) % 8
                else:
                    dstar_path = []
                    action = agent.select_action(state)
            else:
                action = agent.select_action(state)
            state, reward, done = env.step(action)
            step += 1
            traj.append(tuple(env.current_pos))
            last_t = time.time()

            ri = int(np.clip(env.current_pos[0], 0, GH-1))
            ci = int(np.clip(env.current_pos[1], 0, GW-1))
            cur_cls  = class_grid[ri][ci]
            cur_geom = get_geometry(cur_cls)

            # Terrain preview: class of the next cell the rover is heading to
            if dstar_path and len(dstar_path) > 1:
                preview_cls = class_grid[dstar_path[1][0]][dstar_path[1][1]]
            elif theta_path and len(theta_path) > 1:
                next_r = int(np.clip(theta_path[1][0], 0, GH-1))
                next_c = int(np.clip(theta_path[1][1], 0, GW-1))
                preview_cls = class_grid[next_r][next_c]
            else:
                preview_cls = cur_cls

            # Simulate CNN probabilities
            bp = {c: 0.05 for c in NAMES}
            bp[cur_cls] = 0.75
            raw = np.array([bp[c] + random.uniform(0, 0.12) for c in NAMES])
            raw /= raw.sum()
            cur_probs    = raw.tolist()
            cur_cam_conf = max(cur_probs)
            cur_entropy  = min(-sum(p*math.log(p+1e-8) for p in cur_probs)/math.log(4), 1.0)

            u_fused   = compute_fusion(cur_entropy, cur_cam_conf,
                                       cur_geom["lidar_conf"], beta=0.5)
            cur_hscore = compute_hscore(u_fused, cur_cls, alpha=0.75)

            # ── D* Lite replanning on sensor conflict ──────────────────────
            diff = abs(cur_cam_conf - cur_geom["lidar_conf"])
            conflict = diff > 0.35

            if conflict:
                new_cost = min(cost_map[ri, ci] * 2.5 + 0.3, 998)
                cost_map[ri, ci] = new_cost
                dl.update_cell((ri, ci), new_cost)
                # Replan from current rover position to goal
                dstar_path = dl.find_path((ri, ci), goal)
                dstar_active = True
                replan_count += 1
                replan_cells.append((ri, ci))
            else:
                dstar_active = len(dstar_path) > 0

            if done:
                paused = True

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG)

        # Subtle grid background
        for gx in range(0, W, 40):
            pygame.draw.line(screen, (18, 20, 32), (gx, 0), (gx, H), 1)
        for gy in range(0, H, 40):
            pygame.draw.line(screen, (18, 20, 32), (0, gy), (W, gy), 1)

        # Row labels
        screen.blit(font(11, True).render("ROW 1 — SIMULATION", True, DARK_GREY),
                    (PAD, 4))
        screen.blit(font(11, True).render("ROW 2 — SENSOR VIEW", True, DARK_GREY),
                    (PAD, H_ROW+4))

        # Dividers
        pygame.draw.line(screen, BORDER, (0, H_ROW), (W, H_ROW), 1)
        pygame.draw.line(screen, BORDER, (W_LEFT, 0), (W_LEFT, H-H_STATUS), 1)
        pygame.draw.line(screen, BORDER, (W_LEFT+W_MID, 0),
                         (W_LEFT+W_MID, H-H_STATUS), 1)

        p = PAD
        # ROW 1
        r1l = pygame.Rect(p, 18, W_LEFT-2*p, H_ROW-22)
        draw_map_panel(screen, r1l, cost_map, class_grid, traj,
                       theta_path, dstar_path, env.current_pos,
                       goal, step, paused, conflict, replan_cells)

        r1m = pygame.Rect(W_LEFT+p, 18, W_MID-2*p, H_ROW-22)
        draw_cnn_panel(screen, r1m, cur_probs, cur_cam_conf, cur_entropy)

        r1r = pygame.Rect(W_LEFT+W_MID+p, 18, W_RIGHT-2*p, H_ROW-22)
        draw_lidar_panel(screen, r1r, cur_geom, cur_hscore)

        # ROW 2
        r2y = H_ROW + 18
        r2l = pygame.Rect(p, r2y, W_LEFT-2*p, H_ROW-22)
        draw_terrain_panel(screen, r2l, IMGS[preview_cls], preview_cls,
                           cur_entropy, title="  Next Cell Terrain")

        r2m = pygame.Rect(W_LEFT+p, r2y, W_MID-2*p, H_ROW-22)
        draw_camera_panel(screen, r2m, cur_probs, cur_entropy,
                          cur_cam_conf, IMGS[cur_cls])

        r2r = pygame.Rect(W_LEFT+W_MID+p, r2y, W_RIGHT-2*p, H_ROW-22)
        draw_lidar_sensor_panel(screen, r2r, cur_geom, cur_hscore)

        # Status bar
        draw_status(screen,
                    pygame.Rect(0, H-H_STATUS, W, H_STATUS),
                    cur_cls, cur_hscore, cur_cam_conf,
                    cur_geom["lidar_conf"], step,
                    conflict, dstar_active, tick, replan_count)

        # Mission overlay
        if done:
            s = pygame.Surface((W, H), pygame.SRCALPHA)
            s.fill((0, 0, 0, 120))
            screen.blit(s, (0, 0))
            msg = "MISSION COMPLETE!" if reward > 0 else "MISSION FAILED"
            col = GLOW_GREEN if reward > 0 else GLOW_RED
            txt = font(48, True).render(msg, True, col)
            screen.blit(txt, (W//2 - txt.get_width()//2, H//2 - 40))
            sub = font(18).render("Press R to run again", True, WHITE)
            screen.blit(sub, (W//2 - sub.get_width()//2, H//2 + 30))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
