#!/usr/bin/env python3
import os, json
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# CONFIG – keep in sync with train_unet_improved.py
# ----------------------------------------------------------------------
CLASSES = ["background", "wall", "door", "window", "room"]  # 0..4
IMG_SIZE = 640
CKPT_PATH = "runs_unet_improved/best.pt"  # <-- change if your best model lives elsewhere

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# Color map for visualization (BGR)
CLASS_COLORS = {
    "background": (0, 0, 0),
    "wall":       (255, 255, 255),
    "door":       (0, 0, 255),
    "window":     (255, 255, 0),
    "room":       (0, 255, 0),
}


# ----------------------------------------------------------------------
# UNet (same architecture as training, but 5 classes)
# ----------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=len(CLASSES), base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.p4 = nn.MaxPool2d(2)
        self.b = DoubleConv(base * 8, base * 16)

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.c4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        d1 = self.d1(x)
        p1 = self.p1(d1)
        d2 = self.d2(p1)
        p2 = self.p2(d2)
        d3 = self.d3(p2)
        p3 = self.p3(d3)
        d4 = self.d4(p3)
        p4 = self.p4(d4)
        b = self.b(p4)

        u4 = self.u4(b)
        c4 = self.c4(torch.cat([u4, d4], 1))
        u3 = self.u3(c4)
        c3 = self.c3(torch.cat([u3, d3], 1))
        u2 = self.u2(c3)
        c2 = self.c2(torch.cat([u2, d2], 1))
        u1 = self.u1(c2)
        c1 = self.c1(torch.cat([u1, d1], 1))
        return self.out(c1)


def load_model(ckpt: str = CKPT_PATH) -> UNet:
    ck = torch.load(ckpt, map_location=DEVICE)
    classes_ckpt = ck.get("classes", CLASSES)
    if len(classes_ckpt) != len(CLASSES):
        print("[warn] CLASSES in this script differ from checkpoint; using checkpoint list.")
    net = UNet(n_classes=len(classes_ckpt), base=32).to(DEVICE)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


# ----------------------------------------------------------------------
# Pre/post-processing
# ----------------------------------------------------------------------
def preprocess(img: np.ndarray):
    h, w = img.shape[:2]
    scale = IMG_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 255, np.uint8)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas[:nh, :nw] = resized

    x = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5  # normalize to [-1,1]
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return x, (h, w), (nh, nw)


def resize_to_original(mask_small: np.ndarray, orig_hw):
    h, w = orig_hw
    mask = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


# ----------------------------------------------------------------------
# Polygon extraction per class
# ----------------------------------------------------------------------
def mask_to_polygons(mask: np.ndarray, cls_id: int):
    """Extract polygons for a given class ID from mask."""
    m = (mask == cls_id).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for c in cnts:
        if cv2.contourArea(c) < 200:
            continue
        c = cv2.approxPolyDP(c, 2.0, True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in c]
        polys.append(pts)
    return polys


# ----------------------------------------------------------------------
# Main run function
# ----------------------------------------------------------------------
def run(
    img_path: str,
    ckpt: str = CKPT_PATH,
    out_dir: str = "pred_unet_improved",
):
    os.makedirs(out_dir, exist_ok=True)
    net = load_model(ckpt)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    x, orig_hw, _ = preprocess(img)

    with torch.no_grad():
        if DEVICE in ("cuda", "mps"):
            amp_device = "cuda" if DEVICE == "cuda" else "mps"
            with torch.autocast(device_type=amp_device, dtype=torch.float16):
                out = net(x)
        else:
            out = net(x)

    pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)
    mask = resize_to_original(pred, orig_hw)

    # ------------------------------------------------------------------
    # Build polygons JSON for ALL non-background classes
    # ------------------------------------------------------------------
    items = []
    for cid in range(1, len(CLASSES)):  # skip background (0)
        class_name = CLASSES[cid]
        polys = mask_to_polygons(mask, cid)
        for poly in polys:
            items.append({"class": class_name, "points": poly})

    base = Path(img_path).stem
    json_path = os.path.join(out_dir, f"{base}_polygons.json")
    with open(json_path, "w") as f:
        json.dump(items, f)
    print("Wrote polygons JSON:", json_path)

    # ------------------------------------------------------------------
    # Annotated preview on top of original image
    # ------------------------------------------------------------------
    vis = img.copy()
    for it in items:
        cls = it["class"]
        col = CLASS_COLORS.get(cls, (0, 255, 255))
        cv2.polylines(
            vis,
            [np.array(it["points"], np.int32)],
            True,
            col,
            2,
        )
    cv2.imwrite(os.path.join(out_dir, f"{base}_annotated.png"), vis)

    # ------------------------------------------------------------------
    # Save mask as:
    #   - ID mask (0..4)
    #   - colored mask
    # ------------------------------------------------------------------
    # raw ID mask
    cv2.imwrite(os.path.join(out_dir, f"{base}_mask_ids.png"), mask)

    # color mask for visualization
    color_mask = np.zeros((*mask.shape, 3), np.uint8)
    for cid, cname in enumerate(CLASSES):
        color_mask[mask == cid] = CLASS_COLORS.get(cname, (0, 0, 0))
    cv2.imwrite(os.path.join(out_dir, f"{base}_mask_vis.png"), color_mask)

    
def run_folder(folder_path: str, ckpt: str = CKPT_PATH, out_dir: str = "pred_unet_improved"):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    exts = [".jpg", ".jpeg", ".png"]
    imgs = [p for p in folder_path.iterdir() if p.suffix.lower() in exts]

    if not imgs:
        print("No images found in folder:", folder_path)
        return

    print(f"Found {len(imgs)} images. Starting batch inference...\n")

    # Load model once
    net = load_model(ckpt)

    for i, p in enumerate(imgs, 1):
        print(f"[{i}/{len(imgs)}] Processing:", p.name)
        try:
            run(str(p), ckpt=ckpt, out_dir=out_dir)
        except Exception as e:
            print("Error on image", p.name, "->", e)

    print("\nDONE! All outputs saved to:", out_dir)


if __name__ == "__main__":
    # change this to any image you want to test
    #run(
    #    "/Users/senakshikrishnamurthy/Desktop/DS/606 capstone/Sena_PROJECT/FINAL/improved/data/images/test/5_16.jpg"
    #)
    run_folder(
        "/Users/senakshikrishnamurthy/Desktop/DS/606 capstone/Capstone-From-2D-Civil-Floor-Drawings-to-3D-Models/improved/data/images/test",
        ckpt=CKPT_PATH,
        out_dir="pred_unet_improved"
    )
