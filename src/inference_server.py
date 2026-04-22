"""Flask MobileNetV3 terrain inference server. /health, /infer, /batch_infer."""
import base64, json, os, random, sys
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception import TerrainClassifier, run_mc_dropout

NAMES   = ["soil", "bedrock", "sand", "big_rocks"]
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_P = os.path.join(ROOT, "models", "best_model.pt")
LABELS  = os.path.join(ROOT, "data", "good_labels.json")
AI4M    = os.path.join(ROOT, "data", "ai4mars", "ai4mars-dataset-merged-0.1", "msl")
IMG_DIR = os.path.join(AI4M, "images", "edr")
LBL_DIR = os.path.join(AI4M, "labels", "train")
INDEX_P = os.path.join(ROOT, "data", "class_index.json")
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

TFM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_class_index():
    if os.path.exists(INDEX_P):
        return json.load(open(INDEX_P))
    files = json.load(open(LABELS))
    idx = {c: [] for c in NAMES}
    for i, fn in enumerate(files):
        arr  = np.array(Image.open(os.path.join(LBL_DIR, fn)))
        mask = arr[arr != 255]
        if mask.size == 0: continue
        dom = int(np.bincount(mask, minlength=4).argmax())
        if 0 <= dom < 4:
            idx[NAMES[dom]].append(os.path.splitext(fn)[0])
        if (i + 1) % 1000 == 0: print(f"  indexed {i+1}/{len(files)}")
    json.dump(idx, open(INDEX_P, "w"))
    return idx


print(f"Loading model on {DEVICE} ...")
clf   = TerrainClassifier()
model = clf.get_model()
model.load_state_dict(torch.load(MODEL_P, map_location=DEVICE))
model = model.to(DEVICE).eval()

print("Building class index ...")
CLASS_INDEX = build_class_index()
for c in NAMES:
    print(f"  {c}: {len(CLASS_INDEX[c])} images")

app = Flask(__name__)


def infer_one(terrain_class, image_stem=None):
    stems = CLASS_INDEX.get(terrain_class, [])
    if image_stem is None:
        if not stems:
            raise ValueError(f"no images indexed for class {terrain_class}")
        image_stem = random.choice(stems)
    img = Image.open(os.path.join(IMG_DIR, image_stem + ".JPG")).convert("RGB")
    x   = TFM(img).unsqueeze(0).to(DEVICE)
    mean_probs, entropy, _ = run_mc_dropout(model, x, T=20)
    probs = mean_probs.cpu().tolist()
    return {
        "probs":      probs,
        "entropy":    float(entropy.item()),
        "cam_conf":   float(max(probs)),
        "image_stem": image_stem,
    }


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE})


@app.route("/infer", methods=["POST"])
def infer():
    d = request.get_json(force=True)
    return jsonify(infer_one(d["terrain_class"], d.get("image_stem")))


@app.route("/batch_infer", methods=["POST"])
def batch_infer():
    d   = request.get_json(force=True)
    out = [infer_one(c["terrain_class"], c.get("image_stem")) for c in d["cells"]]
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
