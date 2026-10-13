# Two-stage vehicle classification

YOLO detector로 차량 영역을 찾고 별도로 학습한 classifier로 차종을 분류합니다. 원본 이미지 수집, 검사, crop, 분할, 학습과 카메라/영상 추론을 연결합니다.

## 설치

Python 3.11 이상을 사용합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows에서는 `.\.venv\Scripts\Activate.ps1`로 활성화합니다.

## Seed 이미지로 학습

`data/seed_cars/차종이름/사진.jpg` 형태로 독립적인 사진을 클래스마다 두 장 이상 준비합니다. 동일 사진의 복제본은 독립 표본으로 세지 않습니다.

```bash
python run_all.py --mode seed --seed_dir data/seed_cars --device cpu
```

Apple Silicon에서는 `--device mps`, CUDA 환경에서는 `--device cuda:0`을 사용합니다. 옵션을 생략하면 CUDA, MPS, CPU 순으로 선택합니다. 모델의 최초 다운로드에는 인터넷이 필요합니다.

같은 원본은 하나의 분할에만 포함하고, 원본 분할 후 학습 이미지에만 증강을 적용합니다. 재실행은 새 데이터셋을 만들어 교체하며 기존 폴더는 `dataset.previous-...`로 보존합니다. `manifest.json`에 원본 그룹과 분할을 기록합니다.

## 이미 학습한 모델로 추론

```bash
python run_all.py --skip_train --cls_weights runs/classify/car_model/weights/best.pt --source 0 --device cpu
```

`--source`는 카메라 번호 또는 영상 경로입니다. `--no_show`는 창을 띄우지 않습니다. 학습 완료 시 이번 학습의 checkpoint를 사용합니다. 기존 모델 추론 시 checkpoint를 직접 지정해야 합니다.

`--cls_conf` 기본값은 0.6입니다. 그보다 낮은 classifier 점수에는 unknown을 표시합니다. 이 임계값은 미학습 차종을 확실히 구분하는 검증된 방법은 아닙니다. detection 표시를 그리기 전 원본 frame에서 classifier 입력을 잘라냅니다.

## 이미지 검색으로 데이터 준비

`data/queries.txt`에 `클래스명<TAB>검색어`를 적습니다.

```bash
python run_all.py --mode crawl --queries data/queries.txt --per_class 300 --device cpu
```

검색 결과의 라벨은 검색어에서 가져옵니다. 잘못된 차종·동일 차량·유사 사진을 검수해야 합니다. 한 원본의 여러 차량 crop은 같은 source group으로 분할합니다.

## 테스트와 성능 평가

```bash
python -m pip install pytest
python -m pytest -q
```

테스트는 데이터셋 재실행, 원본 그룹 분리, 학습 전용 증강과 device/checkpoint 전달을 확인합니다. 모델 재학습이나 실제 카메라 품질 검증을 대신하지 않습니다.

차종별 정확도·confusion matrix, 독립 촬영 test set, 미학습 차종 오분류와 지연 시간을 별도 평가해야 합니다. 현재 저장소는 이러한 결과를 입증하지 않습니다.
