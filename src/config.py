# ANS Configuration — all hyperparameters in one place

# Grid
CELL_SIZE_M = 0.5        # each cell = 0.5 meters
GRID_SIZE   = 64         # 64x64 grid

# Sensor fusion
MC_PASSES   = 30         # MC Dropout forward passes
BETA        = 0.5        # camera vs lidar fusion weight

# H-score
ALPHA       = 0.5        # uncertainty vs volatility weight
H_CRIT      = 0.6        # impassable threshold

# Cost map weights
W_TERRAIN   = 0.3
W_SLOPE     = 0.2
W_ROUGH     = 0.2
W_HSCORE    = 0.3

# RL
EPISODES    = 500
BATCH_SIZE  = 64
LR          = 1e-4
GAMMA       = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.05

# Terrain volatility
VOLATILITY  = {
    'soil':      0.40,
    'bedrock':   0.10,
    'sand':      0.80,
    'big_rocks': 0.70,
}

# Terrain traversal cost
TERRAIN_COST = {
    'soil':      0.30,
    'bedrock':   0.10,
    'sand':      0.60,
    'big_rocks': 0.90,
}
