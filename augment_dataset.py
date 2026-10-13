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
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f'Cannot write {path}')

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
def build_augmented(src,dst,imgsz=224,val_ratio=.2,multiplier=12,seed=42):
    from dataset_tools import split_sources,stage_directory,publish,clean_stage
    import hashlib
    src,dst=Path(src).resolve(),Path(dst).resolve()
    if src==dst or src in dst.parents or dst in src.parents: raise ValueError('Separate source/output directories required')
    if imgsz<32 or multiplier<0: raise ValueError('Invalid image size/multiplier')
    assignments=split_sources(src,val_ratio,seed)
    random.seed(seed); np.random.seed(seed)
    stage=stage_directory(dst); manifest=[]
    try:
        for path,(cls,split,group) in assignments.items():
            image=imread_rgb(path)
            if image is None: raise ValueError(f'Unreadable image: {path}')
            image=cv2.resize(image,(imgsz,imgsz),interpolation=cv2.INTER_AREA)
            stem=hashlib.sha256(str(path.relative_to(src)).encode()).hexdigest()
            images=[image]+([augment_once(image) for _ in range(multiplier)] if split=='train' else [])
            for i,value in enumerate(images):
                target=stage/split/cls/f'{stem}_{i}.jpg'; imwrite_rgb(target,value)
                manifest.append(dict(source=str(path.relative_to(src)),group=group,split=split,output=str(target.relative_to(stage))))
        return manifest,publish(stage,dst,manifest)
    finally: clean_stage(stage)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--src',default='data/seed'); p.add_argument('--dst',default='data/dataset')
    p.add_argument('--imgsz',type=int,default=224); p.add_argument('--val_ratio',type=float,default=.2)
    p.add_argument('--multiplier',type=int,default=12); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); manifest,backup=build_augmented(a.src,a.dst,a.imgsz,a.val_ratio,a.multiplier,a.seed)
    print(f'Created {len(manifest)} files; previous dataset: {backup}')
if __name__=='__main__': main()
