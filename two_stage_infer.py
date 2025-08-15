#!/usr/bin/env python3
# two_stage_infer.py
# YOLO로 모든 객체 탐지 라벨 표시 + 차량(car/bus/truck)일 때만 2단계 분류 라벨 추가

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# COCO class ids: car=2, bus=5, truck=7
VEHICLE_CLASS_IDS_COCO = {2, 5, 7}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Realtime two-stage: detect everything, classify only vehicles (car/bus/truck)"
    )
    ap.add_argument("--det_weights", type=str, required=True, help="detector weights (e.g., yolov8n.pt)")
    ap.add_argument("--cls_weights", type=str, required=True, help="classifier weights (YOLOv8-CLS best.pt)")
    ap.add_argument("--source", type=str, default="0", help="0/1... for webcam index, or video/rtsp/http path")
    ap.add_argument("--device", type=str, default="mps", help="cpu|mps|cuda (auto-fallback if unavailable)")
    ap.add_argument("--conf", type=float, default=0.35, help="detector confidence threshold")
    ap.add_argument("--show", action="store_true", help="imshow window")
    ap.add_argument("--save", action="store_true", help="save output video")
    ap.add_argument("--save_path", type=str, default="runs/two_stage/out.mp4", help="output video path")
    ap.add_argument("--pad", type=int, default=2, help="padding pixels for crop")
    ap.add_argument("--warmup", type=int, default=3, help="throw away first N frames after opening camera")
    ap.add_argument(
        "--backend",
        type=str,
        choices=["auto", "avf", "default"],
        default="auto",
        help="macOS capture backend: auto/avf=AVFoundation, default=OpenCV default",
    )
    return ap.parse_args()


def move_model_to_device(model: YOLO, device: str):
    try:
        model.to(device)
    except Exception:
        pass  # older versions may not need it / not supported


def select_backend(backend_arg: str):
    """Return cv2 backend int or 0 for default."""
    if sys.platform == "darwin":
        if backend_arg in ("auto", "avf"):
            return cv2.CAP_AVFOUNDATION
        return 0
    return 0


def open_source(src_str: str, backend_arg: str):
    """Open video source (numeric = webcam). On macOS prefer AVFoundation."""
    be = select_backend(backend_arg)
    if src_str.isdigit():
        idx = int(src_str)
        cap = cv2.VideoCapture(idx, be) if be != 0 else cv2.VideoCapture(idx)
    else:
        # for file/rtsp/http we let OpenCV decide
        cap = cv2.VideoCapture(src_str)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src_str}")
    return cap


def crop_box(img, xyxy, pad=2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def main():
    args = parse_args()

    # device sanity
    if args.device == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            print("[WARN] MPS not available; fallback to CPU.")
            args.device = "cpu"

    # load models
    det_model = YOLO(args.det_weights)
    cls_model = YOLO(args.cls_weights)
    move_model_to_device(det_model, args.device)
    move_model_to_device(cls_model, args.device)

    # open source
    cap = open_source(args.source, args.backend)

    # optional writer
    writer = None
    if args.save:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        writer = cv2.VideoWriter(args.save_path, fourcc, fps, (w, h))

    # warmup frames (some webcams return empty first frames)
    for _ in range(max(0, args.warmup)):
        cap.read()

    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # 1) Detection (all classes)
        rlist = det_model.predict(frame, conf=args.conf, device=args.device, verbose=False)
        if not rlist:
            if args.show:
                cv2.imshow("two-stage", frame)
            if writer:
                writer.write(frame)
            if args.show and (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

        r0 = rlist[0]

        # Robust empty-handling (no numpy-vs-list comparison!)
        has_boxes = (r0.boxes is not None and len(r0.boxes) > 0)
        if has_boxes:
            boxes = r0.boxes.xyxy.cpu().numpy()
            cls_ids = r0.boxes.cls.cpu().numpy().astype(int)
            confs = r0.boxes.conf.cpu().numpy()
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            cls_ids = np.empty((0,), dtype=int)
            confs = np.empty((0,), dtype=float)

        names = getattr(r0, "names", None)
        if names is None:
            # fallback if needed
            names = getattr(det_model.model, "names", {})

        det_count = 0

        # 2) Draw all detections; classify vehicles only
        for xyxy, cid, dconf in zip(boxes, cls_ids, confs):
            det_count += 1
            x1, y1, x2, y2 = map(int, xyxy)

            # (A) 항상 YOLO 기본 라벨
            det_name = names.get(int(cid), str(int(cid))) if isinstance(names, dict) else str(int(cid))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                frame,
                f"{det_name} {dconf:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # (B) 차량이면 2단계 분류
            if int(cid) in VEHICLE_CLASS_IDS_COCO:
                crop = crop_box(frame, xyxy, pad=args.pad)
                if crop is None or min(crop.shape[:2]) < 32:
                    continue

                c_res_list = cls_model.predict(crop, device=args.device, verbose=False)
                if not c_res_list:
                    continue
                c0 = c_res_list[0]
                try:
                    probs = c0.probs.data.cpu().numpy()
                    top_idx = int(np.argmax(probs))
                    top_p = float(probs[top_idx])
                    model_name = c0.names.get(top_idx, str(top_idx))
                except Exception:
                    # older API fallback
                    top_idx = int(c0.probs.top1)
                    top_p = float(c0.probs.top1conf)
                    model_name = c0.names.get(top_idx, str(top_idx))

                # 분류 라벨은 감지 라벨 위쪽에 색 다르게 추가
                cv2.putText(
                    frame,
                    f"→ {model_name} {top_p:.2f}",
                    (x1, max(0, y1 - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )

        # HUD: FPS & detection count
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_t))
        prev_t = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}   Det boxes: {det_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if args.show:
            cv2.imshow("two-stage", frame)
        if writer:
            writer.write(frame)

        if args.show and (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
