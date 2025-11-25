
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: 2D Civil Drawings -> TXT of layout measurements (rooms + dimensions)
Author: ChatGPT (starter scaffold)

What it does (MVP):
- Batch scans a folder of 2D drawings (PNG/JPG/TIFF; optional: first page of PDFs)
- Tries to read scale (e.g., 'Scale 1:100' or '1/8"=1\'-0"') via OCR
- Finds dimension text (e.g., 12'-6", 3500 mm, 10'–0") and approximates orientation (H/V)
- Optionally grabs nearby room labels (e.g., 'KITCHEN', 'BEDROOM 2') and groups dimensions by proximity
- Writes per-drawing TXT and a combined JSON Lines with the parsed measurements

Notes:
- This is a scaffold intended to be robust-but-simple. It favors easy wins (dimension strings)
  over full plan vectorization. You can plug in better wall/room detectors later (YOLO/Segment-anything, etc.).
- For PDFs, install pdf2image + poppler or use pdfminer.six to rasterize the first page.
- Requires: Python 3.10+, OpenCV, pytesseract, numpy, shapely (optional but helpful).
"""

import os, re, json, argparse, math, sys, glob
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

# --- Make sure pytesseract can find the Tesseract binary ---
import os, shutil
import pytesseract

# Add common Homebrew paths to PATH for this Python process
paths = ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"]
os.environ["PATH"] = ":".join([*paths, os.environ.get("PATH","")])

# Respect TESSERACT_CMD if you’ve set it, otherwise auto-detect or fall back to common locations
candidates = [
    os.environ.get("TESSERACT_CMD"),
    shutil.which("tesseract"),
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/local/bin/tesseract",
    "/usr/bin/tesseract",
]
tess = next((c for c in candidates if c and os.path.exists(c)), None)
if not tess:
    raise RuntimeError(
        "Tesseract not found. Set TESSERACT_CMD or install it (e.g., brew install tesseract). "
        "If installed, confirm path like /opt/homebrew/bin/tesseract."
    )
pytesseract.pytesseract.tesseract_cmd = tess

# ---- OCR ----
try:
    import pytesseract
    TESS_OK = True
except Exception as e:
    print("[WARN] pytesseract not available; OCR will be skipped:", e, file=sys.stderr)
    TESS_OK = False

DIMENSION_RE = re.compile(
    r"""(?ix)
    (?:~?\b)                              # optional leading tilde/approx
    (\d{1,3}(?:[\.,]\d{1,2})?)         # main number (e.g., 12 or 12.5)
    \s*
    (?:
        (?:['\u2032]                      # feet symbol ( ' or ′ )
            \s* (\d{1,2})?               # optional inches after feet
            (?:\s*["\u2033])?           # inch symbol ( " or ″ ) optional here
        )
        |
        (?:\s*["\u2033])                # inches only (e.g., 6")
        |
        (?:\s*(mm|cm|m))                  # metric
    )?
    (?:\s*(?:-|\\u2013|\\u2014|to)\s*    # optional range hyphen/en dash/em dash/to
       (\d{1,3}(?:[\.,]\d{1,2})?)       # second number for ranges (rare)
       \s*(?:['\u2032]")?               # optional unit hints
    )?
    """
)

SCALE_RE = re.compile(r"""(?ix)
    \b(?:scale\s*[:=]\s*)?
    (?:
        (\d+)\s*[:]\s*(\d+)           # 1:100
      |
        (\d+\/?\d*)\s*"?\s*=?\s*(\d+)'?  # 1/8" = 1'
    )
""")

@dataclass
class Dimension:
    text: str
    value_m: Optional[float]
    orientation: Optional[str]  # 'H' | 'V' | None
    center_xy: Tuple[int, int]
    bbox_xywh: Tuple[int, int, int, int]

@dataclass
class Room:
    name: str
    center_xy: Tuple[int, int]
    bbox_xywh: Tuple[int, int, int, int]

@dataclass
class PlanResult:
    file: str
    scale_text: Optional[str]
    px_per_meter: Optional[float]
    dimensions: List[Dimension]
    rooms: List[Room]

def imread_any(path: str) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # light denoise + adaptive floorplan-friendly binarization
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15)
    return th

def run_ocr(img: np.ndarray, psm=6, oem=3, whitelist=None) -> List[Dict]:
    if not TESS_OK:
        return []
    config = f"--oem {oem} --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
    out = []
    for i in range(len(data['text'])):
        txt = data['text'][i]
        if not txt or txt.strip() == "":
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        conf = float(data['conf'][i]) if data['conf'][i] != '-1' else -1.0
        out.append({"text": txt, "bbox": (x, y, w, h), "conf": conf})
    return out

def try_parse_scale(text_block: str) -> Tuple[Optional[str], Optional[float]]:
    # Return (raw_text, px_per_meter) where px_per_meter is unknown at this stage
    # We only extract the textual descriptor; pixel calibration requires a detected known-length (TODO) or DPI.
    m = SCALE_RE.search(text_block)
    if not m:
        return None, None
    return m.group(0), None

def text_blocks_to_string(ocr: List[Dict]) -> str:
    return " ".join([t["text"] for t in ocr])

def approx_orientation(img: np.ndarray, bbox: Tuple[int,int,int,int]) -> Optional[str]:
    # Use gradients to guess local dominant direction: horizontal vs vertical
    x,y,w,h = bbox
    x0, y0 = max(0,x-10), max(0,y-10)
    x1, y1 = min(img.shape[1], x+w+10), min(img.shape[0], y+h+10)
    roi = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if roi.size == 0:
        return None
    sobelx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    
    sx = float(np.mean(np.abs(sobelx)))
    sy = float(np.mean(np.abs(sobely)))

    if sx > sy * 1.2:
        return 'V'  # more vertical edges nearby -> dimension likely vertical
    if sy > sx * 1.2:
        return 'H'
    return None


def parse_dimension_value_to_m(text: str, px_per_meter: Optional[float]=None) -> Optional[float]:
    # Handles feet/inches and metric. Does not use px_per_meter for text-based dims.
    raw = text.strip().replace("O", "0").replace("o", "0")
    m = DIMENSION_RE.search(raw)
    if not m:
        return None
    num1 = m.group(1)
    inches_after_feet = m.group(2)
    metric_unit = m.group(3)
    num2 = m.group(4)  # for ranges; ignore for MVP

    def feet_inches_to_m(ft: float, inch: float=0.0) -> float:
        return (ft*12.0 + inch) * 0.0254

    # cases
    if metric_unit:
        val = float(num1.replace(",", "."))
        if metric_unit.lower() == 'mm':
            return val / 1000.0
        if metric_unit.lower() == 'cm':
            return val / 100.0
        if metric_unit.lower() == 'm':
            return val
    if "'" in raw or "\u2032" in raw or inches_after_feet is not None:
        ft = float(num1.replace(",", "."))
        inch = float(inches_after_feet) if inches_after_feet else 0.0
        return feet_inches_to_m(ft, inch)
    if '"' in raw or "\u2033" in raw:
        inch = float(num1.replace(",", "."))
        return inch * 0.0254
    # bare number: assume feet if common in plan dims
    try:
        val = float(num1.replace(",", "."))
        # heuristic: treat 6-40 as feet
        if 3.0 <= val <= 60.0:
            return feet_inches_to_m(val, 0.0)
        # otherwise assume meters
        return val
    except:
        return None


def find_dimension_texts(img: np.ndarray, binimg: np.ndarray) -> List[Dimension]:
    # OCR with a dimension-friendly whitelist
    ocr = run_ocr(binimg, psm=6, oem=3, whitelist="0123456789-—–'\\\"cmMftFT.:/ ")
    dims: List[Dimension] = []
    for t in ocr:
        text = t['text']
        if DIMENSION_RE.search(text):
            x,y,w,h = t['bbox']
            orient = approx_orientation(img, (x,y,w,h))
            val_m = parse_dimension_value_to_m(text)
            dims.append(Dimension(text=text, value_m=val_m,
                                  orientation=orient,
                                  center_xy=(x+w//2, y+h//2),
                                  bbox_xywh=(x,y,w,h)))
    return dims

def find_room_labels(binimg: np.ndarray) -> List[Room]:
    # OCR words in ALL CAPS (common for room names)
    ocr = run_ocr(binimg, psm=6, oem=3, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_/")
    rooms: List[Room] = []
    for t in ocr:
        txt = t['text'].strip()
        if len(txt) < 3: 
            continue
        if re.fullmatch(r"[A-Z][A-Z0-9\-/ ]{2,}", txt):
            x,y,w,h = t['bbox']
            rooms.append(Room(name=txt, center_xy=(x+w//2, y+h//2), bbox_xywh=(x,y,w,h)))
    return rooms

def compute_px_per_meter_from_known(img: np.ndarray) -> Optional[float]:
    # TODO: implement ruler/scale bar detection. For now, unknown.
    return None

def process_one(path: str) -> PlanResult:
    img = imread_any(path)
    if img is None:
        return PlanResult(file=path, scale_text=None, px_per_meter=None,
                          dimensions=[], rooms=[])
    binimg = preprocess_for_ocr(img)
    # scale text (no calibration yet)
    ocr_all = run_ocr(binimg, psm=6, oem=3)
    scale_text, _ = try_parse_scale(text_blocks_to_string(ocr_all))
    px_per_m = compute_px_per_meter_from_known(img)

    dims = find_dimension_texts(img, binimg)
    rooms = find_room_labels(binimg)

    return PlanResult(file=path, scale_text=scale_text, px_per_meter=px_per_m,
                      dimensions=dims, rooms=rooms)

def write_txt(out_txt: str, res: PlanResult):
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f"FILE: {os.path.basename(res.file)}\\n")
        f.write(f"SCALE_TEXT: {res.scale_text}\\n")
        f.write(f"PX_PER_METER: {res.px_per_meter}\\n")
        f.write("\\nDIMENSIONS (text -> meters approx; orient H/V):\\n")
        for d in res.dimensions:
            f.write(f" - {d.text:15s} -> {d.value_m if d.value_m else 'NA'} m  orient={d.orientation}\\n")
        f.write("\\nROOM LABELS:\\n")
        for r in res.rooms:
            f.write(f" - {r.name}\\n")

def main():
    ap = argparse.ArgumentParser(description="Batch extract layout measurements from 2D plans -> TXT + JSONL")
    ap.add_argument("--in", dest="in_dir", required=True, help="Folder with PNG/JPG/TIF (and PDFs optional)")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output folder (txt + jsonl)")
    ap.add_argument("--pattern", default="**/*", help="Glob pattern under --in (default '**/*')")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = [p for p in glob.glob(os.path.join(args.in_dir, args.pattern), recursive=True)
             if p.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff','.bmp')) or p.lower().endswith('.pdf')]

    jsonl_path = os.path.join(args.out_dir, "measurements.jsonl")
    jlf = open(jsonl_path, 'w', encoding='utf-8')

    for i, p in enumerate(sorted(paths)):
        print(f"[INFO] ({i+1}/{len(paths)}) Processing: {os.path.basename(p)}")
        res = process_one(p)
        # per-drawing TXT
        out_txt = os.path.join(args.out_dir, f"{os.path.splitext(os.path.basename(p))[0]}_measurements.txt")
        write_txt(out_txt, res)
        # append JSONL
        jlf.write(json.dumps({
            "file": os.path.basename(res.file),
            "scale_text": res.scale_text,
            "px_per_meter": res.px_per_meter,
            "dimensions": [asdict(d) for d in res.dimensions],
            "rooms": [asdict(r) for r in res.rooms],
        }) + "\\n")
    jlf.close()
    print(f"[OK] Wrote JSONL: {jsonl_path}")

if __name__ == "__main__":
    main()
