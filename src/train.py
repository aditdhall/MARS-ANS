import argparse
import json
import os

import torch

from data import get_dataloaders
from perception import TerrainClassifier, get_dominant_label


def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, save_dir, epochs=30):
    model = model.to(device)
    if loss_fn.weight is not None:
        loss_fn.weight = loss_fn.weight.to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            targets = get_dominant_label(labels).to(device)
            logits = model(imgs)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                targets = get_dominant_label(labels).to(device)
                logits = model(imgs)
                loss = loss_fn(logits, targets)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        val_acc = correct / max(total, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    return history


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--good_json', required=True)
    args = parser.parse_args()
    IMG_DIR   = args.img_dir
    LABEL_DIR = args.label_dir
    GOOD_JSON = args.good_json

    train_loader, val_loader, test_loader = get_dataloaders(IMG_DIR, LABEL_DIR, GOOD_JSON)

    tc = TerrainClassifier()
    model = tc.get_model()
    loss_fn = tc.get_loss_fn()
    optimizer = tc.get_optimizer(model)

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(save_dir, exist_ok=True)

    history = train_model(model, train_loader, val_loader, loss_fn, optimizer, device, save_dir)

    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f)
    print("Training complete")
