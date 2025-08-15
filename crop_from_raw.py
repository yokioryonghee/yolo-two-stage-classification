#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

VEHICLE_CIDS = {2,5,7}  # car, bus, truck in COCO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_weights", default="yolov8n.pt")
    ap.add_argument("--src", default="data/clean")
    ap.add_argument("--dst", default="data/crops")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min_side", type=int, default=80)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    model = YOLO(args.det_weights)

    Path(args.dst).mkdir(parents=True, exist_ok=True)

    for cls_dir in Path(args.src).iterdir():
        if not cls_dir.is_dir():
            continue
        out_dir = Path(args.dst)/cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in cls_dir.glob("**/*.jpg"):
            im = cv2.imread(str(p))
            if im is None:
                continue
            res = model.predict(im, conf=args.conf, device=args.device, verbose=False)
            r0 = res[0]
            if not r0.boxes or len(r0.boxes)==0:
                continue
            boxes = r0.boxes.xyxy.cpu().numpy()
            cls_ids = r0.boxes.cls.cpu().numpy().astype(int)
            for i,(x1,y1,x2,y2) in enumerate(boxes):
                if cls_ids[i] not in VEHICLE_CIDS:
                    continue
                x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
                crop = im[max(0,y1):y2, max(0,x1):x2]
                if crop.size==0 or min(crop.shape[:2]) < args.min_side:
                    continue
                cv2.imwrite(str(out_dir/f"{p.stem}_{i}.jpg"), crop)

if __name__ == "__main__":
    main()