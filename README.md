pip install duckduckgo-search(I forgot  👁️👁️)
                                         
                                         
                                         🫦



# Two-Stage Vehicle Pipeline (YOLO Detect + Classify)

This repo provides an **end-to-end two-stage pipeline**:
1) **Detector** (YOLO): uses a pretrained detection model (`--det_weights`, e.g., `yolov8n.pt`) to find objects.
2) https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt

3) **Classifier** (seed-based): classifies **cropped car regions** into fine-grained classes (e.g., model/trim).

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
!The model was trained on a limited dataset, consisting of only a few dozen images for each of the 8 specific car models. While it accurately detects these trained vehicles, it may incorrectly identify or misclassify car types it has not been exposed to. We are currently working on an automated pipeline to streamline the entire process, from data crawling and training to weight updates.


<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/0d8e4bf4-93b6-455c-9d4e-2d6f2f8eab93" />
<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/9be70370-582c-427f-81ea-fbdce3899517" />
<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/afc0ac22-59a0-4dd9-8281-f9a8b2cb48cc" />
<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/d7d22b75-8bdb-404b-a3d4-fef016dc03c0" />
<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/e9cca82f-dab3-4d87-992d-fb00473533f1" />
<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/8f0787a3-c5cb-49fd-a1b8-a64e474a39e4" />
<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/6f3175c2-7b8a-4ed0-96ed-7115045db20d" />
<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/79690d22-198d-497c-8eb6-e30b2c4c51f6" />
<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/3802f565-28be-435b-a049-9fe2290f9dcd" />
<img width="1710" height="1107" alt="Image" src="https://github.com/user-attachments/assets/4a321c39-fad3-429d-9511-2a0a813c6681" />


<img width="3000" height="2250" alt="Image" src="https://github.com/user-attachments/assets/e5700771-e8eb-4cdd-8194-b647bd6b532b" />

<img width="1200" height="1200" alt="Image" src="https://github.com/user-attachments/assets/8d19c3f5-5b1e-40bf-841b-26e1cb0cdab2" />
## License

MIT (change as needed).
