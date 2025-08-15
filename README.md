# Two-Stage Vehicle Pipeline (YOLO Detect + Classify)

This repo provides an **end-to-end two-stage pipeline**:
1) **Detector** (YOLO): uses a pretrained detection model (`--det_weights`, e.g., `yolov8n.pt`) to find objects.
2) **Classifier** (seed-based): classifies **cropped car regions** into fine-grained classes (e.g., model/trim).

The pipeline supports two modes:

- `crawl` mode: web image crawl → verify → detect&crop → split → train classifier → infer
- `seed` mode: **use your own seed images** → augment → train classifier → infer (no crawling)

> The detector is **not retrained** here; only the **classifier** is retrained in `seed` mode.

## Project Tree (key files)
```
run_all.py                # main orchestration (seed/crawl)
augment_dataset.py        # build dataset from seed with augmentations
verify_images.py          # (crawl mode) basic filtering
crop_from_raw.py          # (crawl mode) detect & crop ROIs
split_dataset.py          # create train/val splits
two_stage_infer.py        # real-time 2-stage inference
train_classifier.sh       # trains classifier (Ultralytics classify)
data/
  seed/                   # your seed images (class-per-folder)
  queries.txt             # crawl mode queries
runs/                     # training outputs (ignored by git)
```
![Uploading 스크린샷 2025-07-23 16.27.05.png…]()
## Installation

```bash
# (Recommended) create a venv
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**PyTorch note (macOS MPS/Apple Silicon):** the `torch` wheel installed via `pip` runs with MPS if available (no extra CUDA setup required).

## Quickstart (Seed Mode: no crawling)

1) Prepare seed images (class-per-folder):
```
data/seed_cars/
  Avante/
    img001.jpg ...
  Sonata/
    ...
```

2) Run seed mode (aug → train classifier → infer):
```bash
python run_all.py \
  --mode seed \
  --seed_dir data/seed_cars \
  --imgsz 224 \
  --val_ratio 0.2 \
  --aug_multiplier 12 \
  --det_weights yolov8n.pt \
  --conf 0.35 \
  --source 0
```

After training, the latest classifier checkpoint is auto-picked from `runs/classify/*/weights/best.pt` and used by `two_stage_infer.py` together with the detector.

### Only re-run inference with an existing classifier
```bash
python run_all.py --mode seed --seed_dir data/seed_cars --skip_train --det_weights yolov8n.pt --conf 0.35
```
(Requires an existing `runs/classify/.../best.pt`).

## Crawl Mode (optional, if you want to build data automatically)

```bash
python run_all.py \
  --mode crawl \
  --queries data/queries.txt \
  --per_class 500 \
  --det_weights yolov8n.pt \
  --conf 0.35
```
This will run:
- `crawl_images.py` → `verify_images.py` → `crop_from_raw.py` → `split_dataset.py` → `train_classifier.sh`

To **skip crawling** and still run the rest, prefer **seed mode** with your own `seed_dir`.

## Webcam / iPhone camera input

- `--source` accepts typical OpenCV sources (e.g., `0` for default webcam).
- On macOS with iPhone, Continuity Camera exposes `iPhone Camera` as a webcam-like source in many apps. If you want to stream into this pipeline, you can also use USB solutions like Camo/EpocCam, or Wi‑Fi (NDI/WebRTC) depending on your setup.

## Reproducibility

- Training parameters are logged to `args.yaml` under `runs/classify/<exp-name>/`.
- Typical defaults (from `args.yaml`): `epochs=30`, `batch=32`, `imgsz=224`, `val_ratio=0.2`, `pretrained=true`, `device=mps` (auto).

## Tips

- Keep large artifacts out of Git (see `.gitignore` below).
- Share pretrained weights and sample data via GitHub Releases or external storage.
- For Streamlit UI later, create a new branch:
  ```bash
  git checkout -b feature/streamlit-ui
  ```

## License

MIT (change as needed).
