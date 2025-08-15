#!/usr/bin/env python3
# run_all.py — crawl-mode & seed-mode (augment) unified pipeline

import argparse, subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd, **kw):
    print(f"\n[RUN] {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kw)

def py(args_list):
    return [sys.executable] + args_list

def detect_device():
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def find_script(name: str) -> str:
    # normalize possible trailing-space typo
    bad = ROOT / (name + " ")
    if bad.exists():
        try: bad.rename(ROOT / name)
        except Exception: pass
    for p in [ROOT / name, ROOT / "scripts" / name]:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Script not found: {name}")

def latest_best_cls():
    paths = sorted(ROOT.glob("runs/classify/*/weights/best.pt"), key=os.path.getmtime, reverse=True)
    return str(paths[0]) if paths else ""

def main():
    ap = argparse.ArgumentParser(description="End-to-end pipeline")
    # common
    ap.add_argument("--mode", choices=["crawl","seed"], default="crawl", help="crawl: web pipeline / seed: augment pipeline")
    ap.add_argument("--det_weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None)
    ap.add_argument("--source", default="0")
    ap.add_argument("--skip_train", action="store_true")
    # crawl mode params
    ap.add_argument("--queries", default="data/queries.txt")
    ap.add_argument("--per_class", type=int, default=300)
    ap.add_argument("--crawl_delay", type=float, default=0.6)
    ap.add_argument("--retries", type=int, default=10)
    ap.add_argument("--no_new_patience", type=int, default=5)
    # seed mode params (augment)
    ap.add_argument("--seed_dir", default="data/seed")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--aug_multiplier", type=int, default=12)
    args = ap.parse_args()

    device = args.device or detect_device()
    print(f"[INFO] device = {device}")

    # ensure dirs
    for p in ["data/raw", "data/clean", "data/crops", "data/dataset", "runs"]:
        (ROOT / p).mkdir(parents=True, exist_ok=True)

    if args.mode == "seed":
        # 1) 씨드 → 증강 → dataset 생성
        seed_path = ROOT / args.seed_dir
        if not seed_path.exists() or not any(seed_path.iterdir()):
            print(f"ERROR: seed dir empty: {seed_path}")
            sys.exit(1)

        # (선택) 씨드 검증/정제도 원하면 아래 주석 해제 가능
        # run(py([find_script("verify_images.py"), "--src", args.seed_dir, "--dst", "data/seed_clean"]))
        # seed_used = "data/seed_clean"
        seed_used = args.seed_dir

        run(py([find_script("augment_dataset.py"),
                "--src", seed_used,
                "--dst", "data/dataset",
                "--imgsz", str(args.imgsz),
                "--val_ratio", str(args.val_ratio),
                "--multiplier", str(args.aug_multiplier)]))

        # 2) 학습
        if not args.skip_train:
            sh = Path(find_script("train_classifier.sh"))
            try: sh.chmod(sh.stat().st_mode | 0o111)
            except Exception: pass
            run([str(sh)])

    else:
        # mode == "crawl": 기존 파이프라인
        qfile = ROOT / args.queries
        if not qfile.exists():
            print(f"[WARN] {qfile} not found; creating example Sonata line.")
            (ROOT/"data").mkdir(parents=True, exist_ok=True)
            (ROOT/"data/queries.txt").write_text("Sonata\tHyundai Sonata DN8 exterior front side view\n", encoding="utf-8")
            qfile = ROOT / "data/queries.txt"

        run(py([find_script("crawl_images.py"),
                "--queries", str(qfile),
                "--per_class", str(args.per_class),
                "--delay", str(args.crawl_delay),
                "--retries", str(args.retries),
                "--no_new_patience", str(args.no_new_patience)]))

        run(py([find_script("verify_images.py"), "--src", "data/raw",   "--dst", "data/clean"]))

        run(py([find_script("crop_from_raw.py"),
                "--det_weights", args.det_weights,
                "--src", "data/clean",
                "--dst", "data/crops",
                "--conf", str(args.conf),
                "--device", device]))

        run(py([find_script("split_dataset.py"),
                "--src", "data/crops",
                "--dst", "data/dataset",
                "--val_ratio", str(args.val_ratio)]))

        if not args.skip_train:
            sh = Path(find_script("train_classifier.sh"))
            try: sh.chmod(sh.stat().st_mode | 0o111)
            except Exception: pass
            run([str(sh)])

    # 공통: 최신 분류 best 찾기
    best_cls = latest_best_cls()
    if not best_cls:
        print("ERROR: No classifier best.pt. Check training logs.")
        sys.exit(1)
    print(f"[INFO] classifier best = {best_cls}")

    # 실시간 2단계
    infer = find_script("two_stage_infer.py")
    run(py([infer,
            "--det_weights", args.det_weights,
            "--cls_weights", best_cls,
            "--device", device,
            "--source", args.source,
            "--conf", str(args.conf),
            "--show"]))
    print("\n[DONE] Realtime closed.")

if __name__ == "__main__":
    main()
