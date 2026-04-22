import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_dominant_label(label_tensor):
    B = label_tensor.shape[0]
    result = torch.zeros(B, dtype=torch.long)
    for i in range(B):
        flat = label_tensor[i].flatten()
        valid = flat[flat != 255]
        if len(valid) == 0:
            result[i] = 0
        else:
            result[i] = torch.bincount(valid).argmax()
    return result


class TerrainClassifier:
    def __init__(self):
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(576, 4),
        )

    def get_model(self):
        return self.model

    def get_loss_fn(self):
        weights = torch.tensor([2.7, 2.0, 8.3, 111.0])
        return nn.CrossEntropyLoss(weight=weights)

    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=1e-4)


def run_mc_dropout(model, image_tensor, T=30):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    probs_list = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs)

    mean_probs = torch.stack(probs_list).mean(0).squeeze(0)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(0)
    pred_class = mean_probs.argmax(0)
    return mean_probs, entropy, pred_class


def get_geometry(terrain_class):
    base = {
        "bedrock":   {"slope": 0.07, "roughness": 0.07, "lidar_conf": 0.90},
        "soil":      {"slope": 0.17, "roughness": 0.10, "lidar_conf": 0.85},
        "sand":      {"slope": 0.20, "roughness": 0.20, "lidar_conf": 0.62},
        "big_rocks": {"slope": 0.35, "roughness": 0.60, "lidar_conf": 0.42},
    }[terrain_class]
    slope = float(np.clip(base["slope"] + np.random.normal(0, 0.03), 0.0, 1.0))
    roughness = float(np.clip(base["roughness"] + np.random.normal(0, 0.03), 0.0, 1.0))
    lidar_conf = float(np.clip(base["lidar_conf"] + np.random.normal(0, 0.05), 0.0, 1.0))
    return {"slope": slope, "roughness": roughness, "lidar_conf": lidar_conf}


def compute_fusion(u_mcdropout, cam_conf, lidar_conf, beta=0.5):
    disagreement = abs(cam_conf - lidar_conf)
    u_fused = beta * u_mcdropout + (1 - beta) * disagreement
    return float(u_fused)


def compute_hscore(u_fused, terrain_class, alpha=0.5):
    volatility = {"soil": 0.40, "bedrock": 0.10, "sand": 0.80, "big_rocks": 0.70}
    h_score = alpha * u_fused + (1 - alpha) * volatility[terrain_class]
    return float(h_score)
