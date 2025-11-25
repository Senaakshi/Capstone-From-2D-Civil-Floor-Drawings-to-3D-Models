#!/usr/bin/env python3
"""
Usage:
  python prep_split_and_weaklabels.py \
      --in_dir "/path/to/your/plans" \
      --out_root "./data" \
      --val_ratio 0.15 --test_ratio 0.10 \
      --copy            # omit to use symlinks

It will create:
  data/
    images/{train,val,test}/*.png|jpg
    masks/{train,val}/*.png      # weak masks (0=bg, 1=wall, 4=room)
"""
import os, sys, glob, shutil, random, argparse
from pathlib import Path
import numpy as np
import cv2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def imread_any(p):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def desaturate_and_clean(img):
    """Handles color floorplans (beige fills, green trees) and B/W plans."""
    # Convert to LAB; drop 'a','b' variation to suppress color fills
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # Light bilateral to keep edges, reduce texture
    Lf = cv2.bilateralFilter(L, d=7, sigmaColor=50, sigmaSpace=7)
    return Lf  # grayscale-like channel with good edges

def weak_mask_from_image(img_bgr):
    """
    Returns mask with classes:
      0 background, 1 wall/lines, 4 rooms (interiors)
    """
    L = desaturate_and_clean(img_bgr)

    # Adaptive threshold → binary lines (invert so walls=1)
    bw = cv2.adaptiveThreshold(
        L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=41, C=7
    )

    # Remove tiny specks; thicken lines a bit
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    walls = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    # Seal small gaps so flood-fill can find rooms
    inv = cv2.bitwise_not(walls)
    inv_closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)

    # Connected components → interior blobs as rooms
    num, labels = cv2.connectedComponents(inv_closed)

    mask = np.zeros(L.shape, np.uint8)
    mask[walls > 0] = 1            # walls
    mask[(labels > 0) & (walls == 0)] = 4  # rooms

    # Optional: remove tiny room specks
    room = (mask == 4).astype(np.uint8) * 255
    room = cv2.morphologyEx(room, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    mask[(room == 0) & (mask == 4)] = 0

    return mask

def split_list(paths, val_ratio, test_ratio):
    paths = sorted(paths)
    random.shuffle(paths)
    n = len(paths)
    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))
    test = paths[:n_test]
    val  = paths[n_test:n_test+n_val]
    train = paths[n_test+n_val:]
    return train, val, test

def link_or_copy(src, dst, do_copy=False):
    ensure_dir(os.path.dirname(dst))
    if do_copy:
        shutil.copy2(src, dst)
    else:
        # symlink; fall back to copy on Windows without perms
        try:
            if os.path.exists(dst): os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
        except Exception:
            shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with your *.png/*.jpg plans")
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    args = ap.parse_args()

    imgs = []
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    for p in glob.glob(os.path.join(args.in_dir, "**/*"), recursive=True):
        if p.lower().endswith(exts):
            imgs.append(p)
    if not imgs:
        print("No images found in", args.in_dir); sys.exit(1)

    train, val, test = split_list(imgs, args.val_ratio, args.test_ratio)
    print(f"Found {len(imgs)} images → train={len(train)} val={len(val)} test={len(test)}")

    # Create folder structure
    out_img = { "train": f"{args.out_root}/images/train",
                "val":   f"{args.out_root}/images/val",
                "test":  f"{args.out_root}/images/test" }
    out_msk = { "train": f"{args.out_root}/masks/train",
                "val":   f"{args.out_root}/masks/val" }
    for p in list(out_img.values()) + list(out_msk.values()):
        ensure_dir(p)

    # Place images
    for split, paths in [("train", train), ("val", val), ("test", test)]:
        for src in paths:
            dst = os.path.join(out_img[split], Path(src).name)
            link_or_copy(src, dst, args.copy)

    # Generate weak masks for train/val (not for test)
    for split, paths in [("train", train), ("val", val)]:
        print(f"[{split}] generating weak masks…")
        for src in paths:
            img = imread_any(src)
            mask = weak_mask_from_image(img)
            dst = os.path.join(out_msk[split], Path(src).with_suffix(".png").name)
            cv2.imwrite(dst, mask)

    # Quick sanity counts
    n_empty = 0
    for split in ["train","val"]:
        for ip in glob.glob(os.path.join(out_img[split], "*.*")):
            m = os.path.join(out_msk[split], Path(ip).with_suffix(".png").name)
            if not os.path.exists(m) or os.path.getsize(m)==0:
                n_empty += 1
    print("Done. Empty/missing masks:", n_empty)
    print(f"Data root ready at: {args.out_root}")
    print("Layout:")
    print("  images/train, images/val, images/test")
    print("  masks/train (weak), masks/val (weak)")
    print("\nNext: point your training script to this root.")
if __name__ == "__main__":
    main()
