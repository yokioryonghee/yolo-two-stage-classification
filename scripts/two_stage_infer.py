#!/usr/bin/env python3
import argparse, time
from pathlib import Path
import cv2, numpy as np
import torch
from ultralytics import YOLO

VEHICLE_CLASS_IDS_COCO = {2,5,7}  # car, bus, truck

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_weights", required=True)
    ap.add_argument("--cls_weights", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--save_path", default="runs/two_stage/out.mp4")
    ap.add_argument("--pad", type=int, default=2)
    return ap.parse_args()

def open_source(src):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")
    return cap

def crop(img, xyxy, pad=2):
    h,w = img.shape[:2]
    x1,y1,x2,y2 = map(int, xyxy)
    x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(w-1,x2+pad); y2=min(h-1,y2+pad)
    if x2<=x1 or y2<=y1: return None
    return img[y1:y2, x1:x2].copy()

def main():
    args = parse_args()
    if args.device=="mps" and not torch.backends.mps.is_available():
        print("[WARN] MPS not available → cpu")
        args.device="cpu"
    det = YOLO(args.det_weights)
    clf = YOLO(args.cls_weights)
    cap = open_source(args.source)

    writer=None
    if args.save:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save_path, fourcc, fps, (w,h))

    t0=time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        d = det.predict(frame, conf=args.conf, device=args.device, verbose=False)[0]
        if d.boxes is not None and len(d.boxes)>0:
            boxes = d.boxes.xyxy.cpu().numpy()
            cids  = d.boxes.cls.cpu().numpy().astype(int)
            confs = d.boxes.conf.cpu().numpy()
            for (xyxy,cid,c) in zip(boxes,cids,confs):
                if cid not in VEHICLE_CLASS_IDS_COCO: continue
                cr = crop(frame, xyxy, args.pad)
                if cr is None or min(cr.shape[:2])<32: continue
                r = clf.predict(cr, device=args.device, verbose=False)[0]
                try:
                    probs = r.probs.data.cpu().numpy()
                    k = int(np.argmax(probs)); p=float(probs[k]); name=r.names.get(k,str(k))
                except Exception:
                    k = int(r.probs.top1); p=float(r.probs.top1conf); name=r.names.get(k,str(k))
                x1,y1,x2,y2 = map(int, xyxy)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,180,255),2)
                cv2.putText(frame,f"{name} det:{c:.2f} cls:{p:.2f}",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,180,255),2,cv2.LINE_AA)
        fps = 1.0/max(1e-6, time.time()-t0); t0=time.time()
        cv2.putText(frame,f"FPS:{fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
        if args.show: cv2.imshow("two-stage", frame)
        if writer: writer.write(frame)
        if args.show and (cv2.waitKey(1)&0xFF)==27: break

    cap.release();
    if writer: writer.release()
    if args.show: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()