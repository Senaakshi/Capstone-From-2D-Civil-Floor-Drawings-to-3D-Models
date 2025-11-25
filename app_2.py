#!/usr/bin/env python3
import os, json
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, request, render_template, send_from_directory

# -----------------------------
# Import pipelines
# -----------------------------
# PIPELINE A: improved UNet + coloured 3D
from infer_unet_improved import (
    preprocess as preprocess_A,
    resize_to_original as resize_A,
    mask_to_polygons as mask_to_polygons_A,
    CLASSES as CLASSES_A,
    DEVICE as DEVICE_A,
    load_model as load_model_A,
)

# PIPELINE B: second UNet + different style 3D
from infer_unet import (
    preprocess as preprocess_B,
    resize_to_original as resize_B,
    mask_to_polygons as mask_to_polygons_B,
    CLASSES as CLASSES_B,
    DEVICE as DEVICE_B,
    load_model as load_model_B,
)

from to3d_improved_colour import build_mesh as build_mesh_A
from to3d_colour import build_mesh as build_mesh_B

# -----------------------------
# Paths & folders
# -----------------------------
BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "web_uploads"
PRED_DIR   = BASE_DIR / "web_pred"
OUT_3D_DIR = BASE_DIR / "web_out_3d"

for d in (UPLOAD_DIR, PRED_DIR, OUT_3D_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load both UNets once
# -----------------------------
print("Loading UNet models...")

unet_A = load_model_A()   # uses CKPT_PATH inside infer_unet_improved.py
unet_A.eval()

unet_B = load_model_B()   # uses CKPT_PATH (or similar) inside infer_unet.py
unet_B.eval()

print("Models loaded. Devices:", DEVICE_A, DEVICE_B)

app = Flask(__name__)

# -----------------------------
# Inference helpers
# -----------------------------
def run_unet_A(img: np.ndarray, stem: str):
    """
    Run improved UNet (pipeline A) on an image and save polygons_A.json
    """
    polygons_json_A = PRED_DIR / f"{stem}_A_polygons.json"

    x, orig_hw, _ = preprocess_A(img)

    with torch.no_grad():
        if DEVICE_A in ("cuda", "mps"):
            amp_device = "cuda" if DEVICE_A == "cuda" else "mps"
            with torch.autocast(device_type=amp_device, dtype=torch.float16):
                out = unet_A(x)
        else:
            out = unet_A(x)

    pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)
    mask = resize_A(pred, orig_hw)

    items = []
    for cid in range(1, len(CLASSES_A)):  # skip background
        class_name = CLASSES_A[cid]
        polys = mask_to_polygons_A(mask, cid)
        for poly in polys:
            items.append({"class": class_name, "points": poly})

    polygons_json_A.parent.mkdir(parents=True, exist_ok=True)
    with open(polygons_json_A, "w") as f:
        json.dump(items, f)

    print("Wrote polygons A:", polygons_json_A)
    return polygons_json_A


def run_unet_B(img: np.ndarray, stem: str):
    """
    Run second UNet (pipeline B) on an image and save polygons_B.json
    """
    polygons_json_B = PRED_DIR / f"{stem}_B_polygons.json"

    x, orig_hw, _ = preprocess_B(img)

    with torch.no_grad():
        if DEVICE_B in ("cuda", "mps"):
            amp_device = "cuda" if DEVICE_B == "cuda" else "mps"
            with torch.autocast(device_type=amp_device, dtype=torch.float16):
                out = unet_B(x)
        else:
            out = unet_B(x)

    pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)
    mask = resize_B(pred, orig_hw)

    items = []
    for cid in range(1, len(CLASSES_B)):  # skip background
        class_name = CLASSES_B[cid]
        polys = mask_to_polygons_B(mask, cid)
        for poly in polys:
            items.append({"class": class_name, "points": poly})

    polygons_json_B.parent.mkdir(parents=True, exist_ok=True)
    with open(polygons_json_B, "w") as f:
        json.dump(items, f)

    print("Wrote polygons B:", polygons_json_B)
    return polygons_json_B


def run_two_inferences(img_path: str, stem: str):
    """
    Convenience: read image once, run both models, return both json paths.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    polyA = run_unet_A(img, stem)
    polyB = run_unet_B(img, stem)
    return polyA, polyB


def build_two_models(polyA: Path, polyB: Path, stem: str):
    """
    From polygons_A + polygons_B → two GLBs with different 3D styles.
    """
    glb_A = OUT_3D_DIR / f"{stem}_A.glb"
    glb_B = OUT_3D_DIR / f"{stem}_B.glb"

    build_mesh_A(str(polyA), str(glb_A))  # coloured / improved style
    build_mesh_B(str(polyB), str(glb_B))  # second style

    print("Built GLB A:", glb_A)
    print("Built GLB B:", glb_B)
    return glb_A, glb_B

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "plan" not in request.files:
        return "No file part", 400

    file = request.files["plan"]
    if file.filename == "":
        return "No selected file", 400

    filename = file.filename
    stem = Path(filename).stem
    img_path = UPLOAD_DIR / filename
    file.save(img_path)

    # 1) Run BOTH UNet pipelines
    polyA, polyB = run_two_inferences(str(img_path), stem)

    # 2) Build TWO 3D models (different styles)
    glbA, glbB = build_two_models(polyA, polyB, stem)

    # 3) Render split-view page with both models
    return render_template(
        "result.html",
        name=stem,
        modelA=glbA.name,
        modelB=glbB.name,
    )


@app.route("/models/<path:filename>")
def serve_model(filename):
    # Serve GLB models to the browser
    return send_from_directory(OUT_3D_DIR, filename)


# Auto-open browser
import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    # Open browser after 1.5 seconds
    Timer(1.5, open_browser).start()

    # Start Flask
    app.run(debug=True)
