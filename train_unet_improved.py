#!/usr/bin/env python3
# train_unet.py  — compact, MPS-friendly, warning-free

import os, glob
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# -------------------- Config --------------------
CLASSES     = ["background", "wall", "door", "window", "room"]  # 0=bg,1=wall,2=door,3=window,4=room
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 768         # smaller -> less MPS memory (try 768 if you can)
BATCH       = 1            # keep 1 on MPS to avoid OOM
EPOCHS      = 60
LR          = 1e-3
ROOT        = "improved/data"       # expect data/images/{train,val}, data/masks/{train,val}
SAVE_DIR    = "runs_unet_improved"

DEVICE = "mps" if torch.backends.mps.is_available() else \
         ("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)
# -------------------- Dataset --------------------
class PlanSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.mask_dir = mask_dir
        self.augment = augment

        # Albumentations: use current API (no value/mask_value on Pad/ShiftScaleRotate)
        self.t_train = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                          border_mode=cv2.BORDER_CONSTANT),
            A.Affine(scale=(0.95, 1.05),
                     translate_percent=(0.0, 0.02),
                     rotate=(-3, 3),
                     mode=cv2.BORDER_CONSTANT,
                     cval=255,      # image fill
                     cval_mask=0,   # mask fill
                     p=0.7),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(0.05, 0.05, p=0.3),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2()
        ])
        self.t_val = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                          border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2()
        ])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        ip = self.imgs[idx]
        name = Path(ip).stem
        mp = os.path.join(self.mask_dir, f"{name}.png")
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None:  raise FileNotFoundError(ip)
        if mask is None: raise FileNotFoundError(mp)

        # enforce label range
        mask = np.clip(mask, 0, NUM_CLASSES-1).astype(np.uint8)

        aug = (self.t_train if self.augment else self.t_val)(image=img, mask=mask)
        x   = aug["image"]                 # CxHxW float32
        y   = aug["mask"].long().squeeze() # HxW long
        return x, y

# -------------------- Model (lean U-Net) --------------------
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
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=NUM_CLASSES, base=32):  # base=32 to save memory
        super().__init__()
        self.d1 = DoubleConv(in_ch, base);    self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2);   self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4); self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base*4, base*8); self.p4 = nn.MaxPool2d(2)
        self.b  = DoubleConv(base*8, base*16)
        self.u4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.c4 = DoubleConv(base*16, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.c3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.c2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.c1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        d4 = self.d4(p3); p4 = self.p4(d4)
        b  = self.b(p4)
        u4 = self.u4(b);  c4 = self.c4(torch.cat([u4, d4], dim=1))
        u3 = self.u3(c4); c3 = self.c3(torch.cat([u3, d3], dim=1))
        u2 = self.u2(c3); c2 = self.c2(torch.cat([u2, d2], dim=1))
        u1 = self.u1(c2); c1 = self.c1(torch.cat([u1, d1], dim=1))
        return self.out(c1)

# -------------------- Loss & Metrics --------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps = eps
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        onehot = torch.nn.functional.one_hot(targets, logits.shape[1]).permute(0,3,1,2).float()
        inter = (probs * onehot).sum((0,2,3))
        den   = (probs + onehot).sum((0,2,3))
        dice  = (2*inter + self.eps) / (den + self.eps)
        return 1 - dice.mean()

def mean_iou(logits, targets, n_classes):
    pred = torch.argmax(logits, dim=1)
    ious=[]; 
    for c in range(n_classes):
        p = (pred==c); t = (targets==c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union>0: ious.append(inter/union)
    return float(np.mean(ious)) if ious else 0.0

# -------------------- Train --------------------
def main():
    train_ds = PlanSegDataset(f"{ROOT}/images/train", f"{ROOT}/masks/train", augment=True)
    val_ds   = PlanSegDataset(f"{ROOT}/images/val",   f"{ROOT}/masks/val",   augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)

    net = UNet(n_classes=NUM_CLASSES, base=32).to(DEVICE)
    ce   = nn.CrossEntropyLoss()
    dice = DiceLoss()
    opt  = torch.optim.AdamW(net.parameters(), lr=LR)

    best_miou, best_path = 0.0, os.path.join(SAVE_DIR, "best.pt")
    use_amp = (DEVICE in ["mps","cuda"])

    for epoch in range(1, EPOCHS+1):
        net.train(); tl = 0.0
        for x,y in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type=("cuda" if DEVICE=="cuda" else "mps"),
                                dtype=torch.float16, enabled=use_amp):
                out  = net(x)
                loss = 0.7*ce(out,y) + 0.3*dice(out,y)
            loss.backward(); opt.step()
            tl += loss.item()

        # ---- Validation
        net.eval(); vl=0.0; miou=0.0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(DEVICE), y.to(DEVICE)
                with torch.autocast(device_type=("cuda" if DEVICE=="cuda" else "mps"),
                                    dtype=torch.float16, enabled=use_amp):
                    out = net(x)
                    vl += (0.7*ce(out,y)+0.3*dice(out,y)).item()
                miou += mean_iou(out.float(), y, NUM_CLASSES)
        miou /= max(1,len(val_dl))

        print(f"[E{epoch:02d}] train_loss={tl/len(train_dl):.4f}  val_loss={vl/len(val_dl):.4f}  mIoU={miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save({"model": net.state_dict(),
                        "classes": CLASSES,
                        "img_size": IMG_SIZE}, best_path)
            print(f"   ↳ saved best to {best_path} (mIoU={best_miou:.4f})")

if __name__ == "__main__":
    main()
