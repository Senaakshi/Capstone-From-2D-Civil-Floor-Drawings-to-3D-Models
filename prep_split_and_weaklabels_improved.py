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
    masks/{train,val}/*.png      # weak masks:
                                  # 0 = background
                                  # 1 = walls (lines / shell)
                                  # 4 = rooms (interiors)

NOTE:
  - This script does NOT auto-detect doors/windows.
  - Classes 2 (door) and 3 (window) are reserved for
    future/manual/YOLO annotations.
"""

import os, sys, glob, shutil, random, argparse
from pathlib import Path
import numpy as np
import cv2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def imread_any(p: str) -> np.ndarray:
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def link_or_copy(src: str, dst: str, do_copy: bool = False):
    ensure_dir(os.path.dirname(dst))
    if do_copy:
        shutil.copy2(src, dst)
    else:
        # symlink; fall back to copy on Windows without perms
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
        except Exception:
            shutil.copy2(src, dst)

# -----------------------------------------------------------------------------
# Weak mask generation (0,1,4 labels)
# -----------------------------------------------------------------------------

def desaturate_and_clean(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert color floorplans (beige fills, green trees) or B/W plans
    into a single-channel image with strong, clean edges.

    Returns:
        Lf: uint8, same HxW, "lightness" channel filtered.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # Light bilateral to keep edges, reduce texture/noise
    Lf = cv2.bilateralFilter(L, d=7, sigmaColor=50, sigmaSpace=7)
    return Lf

def weak_mask_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Build a WEAK mask from a raw floorplan.

    Output mask (uint8) uses:
      0 = background
      1 = walls / linework (outer & inner)
      4 = room interiors

    Classes 2 and 3 (door/window) are intentionally NOT created here,
    because they usually require explicit annotation or a specialized
    detector. They will be 0 (background) in these weak masks.
    """
    L = desaturate_and_clean(img_bgr)

    # --- 1) Get wall/line candidates via adaptive threshold
    # Invert so walls/lines ≈ white (255), background ≈ 0
    bw = cv2.adaptiveThreshold(
        L, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=41,
        C=7
    )

    # --- 2) Remove tiny specks; thicken lines
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    # --- 3) Prepare for interior flood-fill: close gaps in "background"
    inv = cv2.bitwise_not(walls)
    inv_closed = cv2.morphologyEx(
        inv,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2
    )

    # --- 4) Connected components on closed background → interior blobs
    num_labels, labels = cv2.connectedComponents(inv_closed)

    # --- 5) Build label mask
    mask = np.zeros(L.shape, np.uint8)

    # walls (any pixel with wall line)
    mask[walls > 0] = 1

    # rooms: connected components where there is no wall
    mask[(labels > 0) & (walls == 0)] = 4

    # --- 6) Remove tiny room specks (morph open)
    room_bin = (mask == 4).astype(np.uint8) * 255
    room_bin = cv2.morphologyEx(
        room_bin,
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8),
        iterations=1
    )
    # Where we previously marked 4 but morph-open removed it → set back to 0
    mask[(room_bin == 0) & (mask == 4)] = 0

    # Safety: ensure labels are within [0,4]
    mask = np.clip(mask, 0, 4).astype(np.uint8)

    return mask

# -----------------------------------------------------------------------------
# Train/val/test split
# -----------------------------------------------------------------------------

def split_list(paths, val_ratio, test_ratio):
    paths = sorted(paths)
    random.shuffle(paths)
    n = len(paths)
    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))
    test = paths[:n_test]
    val  = paths[n_test:n_test + n_val]
    train = paths[n_test + n_val:]
    return train, val, test

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True,
                    help="Folder with your *.png/*.jpg plans")
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of symlink")
    args = ap.parse_args()

    # Collect images
    imgs = []
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    for p in glob.glob(os.path.join(args.in_dir, "**/*"), recursive=True):
        if p.lower().endswith(exts):
            imgs.append(p)

    if not imgs:
        print("No images found in", args.in_dir)
        sys.exit(1)

    train, val, test = split_list(imgs, args.val_ratio, args.test_ratio)
    print(f"Found {len(imgs)} images → "
          f"train={len(train)} val={len(val)} test={len(test)}")

    # Create folder structure
    out_img = {
        "train": f"{args.out_root}/images/train",
        "val":   f"{args.out_root}/images/val",
        "test":  f"{args.out_root}/images/test",
    }
    out_msk = {
        "train": f"{args.out_root}/masks/train",
        "val":   f"{args.out_root}/masks/val",
    }
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
            dst = os.path.join(
                out_msk[split],
                Path(src).with_suffix(".png").name
            )
            cv2.imwrite(dst, mask)

    # Quick sanity check: unique labels per split
    for split in ["train", "val"]:
        uniq_all = set()
        for ip in glob.glob(os.path.join(out_img[split], "*.*")):
            mpath = os.path.join(
                out_msk[split],
                Path(ip).with_suffix(".png").name
            )
            if not os.path.exists(mpath):
                continue
            m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            uniq = np.unique(m)
            uniq_all.update(uniq.tolist())
        print(f"[{split}] unique labels across masks: {sorted(uniq_all)}")

    print(f"\nDone. Data root ready at: {args.out_root}")
    print("Layout:")
    print("  images/train, images/val, images/test")
    print("  masks/train (weak), masks/val (weak)")
    print("Classes present in these masks: 0=bg, 1=wall, 4=room")
    print("Classes 2 (door) & 3 (window) are reserved for future annotations.")

if __name__ == "__main__":
    main()
