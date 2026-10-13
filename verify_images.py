#!/usr/bin/env python3
import os, argparse
from pathlib import Path
from dataset_tools import stage_directory,publish
from PIL import Image
import imagehash

MIN_SIDE = 224

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--dst", default="data/clean")
    args = ap.parse_args()

    final_dst=Path(args.dst).resolve()
    source=Path(args.src).resolve()
    if source==final_dst or source in final_dst.parents or final_dst in source.parents:
        raise ValueError('Separate source/output directories required')
    stage=stage_directory(final_dst)
    args.dst=str(stage)

    for cls_dir in Path(args.src).iterdir():
        if not cls_dir.is_dir():
            continue
        hashes = set()
        out_dir = Path(args.dst)/cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for p in cls_dir.glob("**/*"):
            if not p.suffix.lower() in [".jpg",".jpeg",".png"]:
                continue
            try:
                im = Image.open(p).convert("RGB")
                if min(im.size) < MIN_SIDE:
                    continue
                h = imagehash.average_hash(im)
                if h in hashes:
                    continue
                hashes.add(h)
                im.save(out_dir/(p.stem+".jpg"), quality=92)
                kept += 1
            except Exception:
                pass
        print(f"[verify] {cls_dir.name}: kept {kept}")

    publish(stage,final_dst,{'source':str(source)})

if __name__ == "__main__":
    main()
