import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import config
from perception import compute_fusion, compute_hscore, get_geometry, run_mc_dropout


class AI4MarsDataset(Dataset):
    def __init__(self, img_dir, label_dir, good_labels_list, transform):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.stems = [Path(n).stem for n in good_labels_list]
        self.transform = transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img = Image.open(os.path.join(self.img_dir, stem + ".JPG")).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, stem + ".png"))
        label = label.resize((256, 256), Image.NEAREST)
        img = self.transform(img)
        label = torch.from_numpy(np.array(label)).long()
        return img, label


def get_dataloaders(img_dir, label_dir, good_labels_json_path, batch_size=16):
    with open(good_labels_json_path) as f:
        good_labels_list = json.load(f)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    n = len(good_labels_list)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(42)
    train_idx, val_idx, test_idx = random_split(range(n), [n_train, n_val, n_test], generator=gen)

    train_files = [good_labels_list[i] for i in train_idx]
    val_files = [good_labels_list[i] for i in val_idx]
    test_files = [good_labels_list[i] for i in test_idx]

    train_ds = AI4MarsDataset(img_dir, label_dir, train_files, train_tf)
    val_ds = AI4MarsDataset(img_dir, label_dir, val_files, eval_tf)
    test_ds = AI4MarsDataset(img_dir, label_dir, test_files, eval_tf)

    class_weights = torch.tensor([2.7, 2.0, 8.3, 111.0])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Class weights (soil, bedrock, sand, big_rocks): {class_weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def build_cost_map(terrain_grid, model, device, alpha=0.5, beta=0.5, h_crit=0.6):
    H = len(terrain_grid)
    W = len(terrain_grid[0])
    cost_map = np.zeros((H, W), dtype=float)
    model = model.to(device)
    for i in range(H):
        for j in range(W):
            cell = terrain_grid[i][j]
            terrain_class = cell["terrain_class"]
            img_tensor = cell["img_tensor"].to(device)
            mean_probs, entropy, _ = run_mc_dropout(model, img_tensor, T=30)
            cam_conf = mean_probs.max().item()
            geom = get_geometry(terrain_class)
            u_fused = compute_fusion(entropy.item(), cam_conf, geom["lidar_conf"], beta)
            h_score = compute_hscore(u_fused, terrain_class, alpha)
            if h_score > h_crit:
                cost_map[i, j] = 999
            else:
                terrain_cost = config.TERRAIN_COST[terrain_class]
                cost_map[i, j] = (
                    0.3 * terrain_cost
                    + 0.2 * geom["slope"]
                    + 0.2 * geom["roughness"]
                    + 0.3 * h_score
                )
    return cost_map
