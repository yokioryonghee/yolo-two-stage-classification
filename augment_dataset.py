#!/usr/bin/env python3
# augment_dataset.py — seed -> dataset(train/val), heavy augment for train only

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

# -----------------------------
# IO helpers
# -----------------------------
def imread_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imwrite_rgb(path: Path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

# -----------------------------
# Augmentations tuned for webcam-like noise
# -----------------------------
def rand_motion_blur(img):
    k = random.choice([3, 5, 7, 9])
    kernel = np.zeros((k, k), dtype=np.float32)
    if random.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)

def rand_perspective(img):
    h, w = img.shape[:2]
    d = int(0.08 * min(h, w))
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [random.randint(0, d), random.randint(0, d)],
        [w - random.randint(0, d), random.randint(0, d)],
        [random.randint(0, d), h - random.randint(0, d)],
        [w - random.randint(0, d), h - random.randint(0, d)],
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def rand_color(img):
    pil = Image.fromarray(img)
    if random.random() < 0.9:
        pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.6, 1.4))
    if random.random() < 0.9:
        pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.6, 1.4))
    if random.random() < 0.8:
        pil = ImageEnhance.Color(pil).enhance(random.uniform(0.6, 1.4))
    return np.array(pil)

def rand_noise(img):
    if random.random() < 0.8:
        sigma = random.uniform(3, 10)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        x = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return x
    return img

def rand_jpeg(img):
    q = random.randint(40, 80)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

def rand_cutout(img):
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(random.randint(0, 2)):
        ch = random.randint(max(1, h // 12), max(2, h // 6))
        cw = random.randint(max(1, w // 12), max(2, w // 6))
        y = random.randint(0, max(0, h - ch))
        x = random.randint(0, max(0, w - cw))
        out[y:y + ch, x:x + cw] = random.randint(0, 30)
    return out

def augment_once(img):
    x = img
    if random.random() < 0.5: x = rand_motion_blur(x)
    if random.random() < 0.6: x = rand_perspective(x)
    x = rand_color(x)
    x = rand_noise(x)
    if random.random() < 0.7: x = rand_jpeg(x)
    if random.random() < 0.5: x = rand_cutout(x)
    return x

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/seed", help="seed images per class")
    ap.add_argument("--dst", default="data/dataset", help="output dataset (ImageFolder)")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--multiplier", type=int, default=12, help="augmented images per seed train image")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    (dst / "train").mkdir(parents=True, exist_ok=True)
    (dst / "val").mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in src.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No classes found in {src}")

    rng = random.Random(42)

    for cls in sorted(classes):
        src_dir = src / cls
        files = [p for p in src_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        rng.shuffle(files)

        if not files:
            print(f"[{cls}] no seed images, skip.")
            continue

        k = int(len(files) * (1 - args.val_ratio))
        k = max(1, min(k, len(files) - 1)) if len(files) >= 2 else 1  # ensure both split have at least 1 if possible
        train_files = files[:k]
        val_files = files[k:]

        # val: copy resized, no augmentation
        val_out = dst / "val" / cls
        val_out.mkdir(parents=True, exist_ok=True)
        v_ok = 0
        for p in val_files:
            im = imread_rgb(p)
            if im is None:
                continue
            im = cv2.resize(im, (args.imgsz, args.imgsz), interpolation=cv2.INTER_AREA)
            imwrite_rgb(val_out / f"{p.stem}.jpg", im)
            v_ok += 1

        # train: original + augmented N
        train_out = dst / "train" / cls
        train_out.mkdir(parents=True, exist_ok=True)
        t_ok = 0
        for p in train_files:
            im0 = imread_rgb(p)
            if im0 is None:
                continue
            im0 = cv2.resize(im0, (args.imgsz, args.imgsz), interpolation=cv2.INTER_AREA)
            imwrite_rgb(train_out / f"{p.stem}_orig.jpg", im0)
            t_ok += 1
            for i in range(args.multiplier):
                imi = augment_once(im0)
                imwrite_rgb(train_out / f"{p.stem}_a{i}.jpg", imi)
                t_ok += 1

        print(f"[{cls}] seed:{len(files)}  -> train:{t_ok}  val:{v_ok}")

if __name__ == "__main__":
    main()
