#!/usr/bin/env python3
import os, random, shutil, argparse
from pathlib import Path

random.seed(42)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/crops")
    ap.add_argument("--dst", default="data/dataset")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    args = ap.parse_args()

    for split in ["train","val"]:
        Path(args.dst, split).mkdir(parents=True, exist_ok=True)

    for cls_dir in Path(args.src).iterdir():
        if not cls_dir.is_dir():
            continue
        imgs = [p for p in cls_dir.glob("**/*.jpg")]
        if len(imgs)==0:
            continue
        random.shuffle(imgs)
        k = int(len(imgs)*(1.0-args.val_ratio))
        train, val = imgs[:k], imgs[k:]
        for name, arr in [("train",train),("val",val)]:
            out = Path(args.dst, name, cls_dir.name)
            out.mkdir(parents=True, exist_ok=True)
            for p in arr:
                shutil.copy2(p, out/p.name)
        print(f"[split] {cls_dir.name}: train {len(train)} / val {len(val)}")

if __name__ == "__main__":
    main()