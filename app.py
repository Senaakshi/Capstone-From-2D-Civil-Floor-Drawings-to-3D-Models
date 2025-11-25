#!/usr/bin/env python3
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, request, render_template, send_from_directory

# Import pieces from your existing scripts
from infer_unet_improved import (
    preprocess,
    resize_to_original,
    mask_to_polygons,
    CLASSES,
    DEVICE,
    load_model,
)
from to3d_improved_colour import build_mesh


# -----------------------------
# Paths & folders
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "web_uploads"
PRED_DIR   = BASE_DIR / "web_pred"
OUT_3D_DIR = BASE_DIR / "web_out_3d"

for d in (UPLOAD_DIR, PRED_DIR, OUT_3D_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load UNet ONCE on server start
# -----------------------------
print("Loading UNet model...")
unet = load_model()   # uses CKPT_PATH inside infer_unet_improved.py
unet.eval()
print("Model loaded, device:", DEVICE)

app = Flask(__name__)


# -----------------------------
# Helper: run UNet on ONE image
# -----------------------------
def run_unet_inference_single(img_path: str, polygons_json_path: str):
    """
    Run your improved UNet on a single image and write polygons.json
    in the same format as infer_unet_improved.run().
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    x, orig_hw, _ = preprocess(img)

    with torch.no_grad():
        if DEVICE in ("cuda", "mps"):
            amp_device = "cuda" if DEVICE == "cuda" else "mps"
            with torch.autocast(device_type=amp_device, dtype=torch.float16):
                out = unet(x)
        else:
            out = unet(x)

    pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)
    mask = resize_to_original(pred, orig_hw)

    # Build polygons list (same logic as in infer_unet_improved.run)
    items = []
    for cid in range(1, len(CLASSES)):  # skip background (0)
        class_name = CLASSES[cid]
        polys = mask_to_polygons(mask, cid)
        for poly in polys:
            items.append({"class": class_name, "points": poly})

    # Ensure folder exists
    polygons_json_path = Path(polygons_json_path)
    polygons_json_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(polygons_json_path, "w") as f:
        json.dump(items, f)

    print("Wrote polygons JSON:", polygons_json_path)


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

    # Save uploaded image
    filename = file.filename
    stem = Path(filename).stem
    img_path = UPLOAD_DIR / filename
    file.save(img_path)

    # Derive output paths
    polygons_json = PRED_DIR / f"{stem}_polygons.json"
    glb_path = OUT_3D_DIR / f"{stem}.glb"

    # 1) Run UNet → polygons.json
    run_unet_inference_single(str(img_path), str(polygons_json))

    # 2) Build 3D GLB from polygons.json
    build_mesh(str(polygons_json), str(glb_path))

    # 3) Show result page with embedded 3D viewer
    return render_template("result.html", model_name=f"{stem}.glb")


@app.route("/models/<path:filename>")
def serve_model(filename):
    # Serve GLB models to the browser
    return send_from_directory(OUT_3D_DIR, filename)

import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    # Open browser after 1.5 seconds
    Timer(1.5, open_browser).start()

    # Start Flask
    app.run(debug=True)

#if __name__ == "__main__":
    # For local testing
    #app.run(debug=True)
    