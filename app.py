# app.py
import os
import time
import threading
from pathlib import Path
from typing import Dict, Tuple, Optional

import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# ======================
# CONFIG (수정 포인트)
# ======================
DETECT_WEIGHTS = "yolov8n.pt"          # Ultralytics 기본 탐지 가중치
CLASSIFY_WEIGHTS = "yolov8n-cls.pt"    #튜닝된 가중치 파일

# (처음 실행 시 자동 다운로드할 URL) — 아래 2개 중 최소 C L S 는 꼭 채우기
WEIGHTS_URL_DET = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
WEIGHTS_URL_CLS = "https://github.com/yokioryonghee/yolo-two-stage-classification/releases/download/yolov8n-cls.pt/yolov8n-cls.pt"  # ← 너의 Release URL로 바꾸기

CONF_THRES = 0.35
TARGET_CLASSES = {"car", "truck", "bus"}  # 탐지 단계에서 분류기로 보낼 대상

# WebRTC (TURN 권장)
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        # {"urls": ["turn:YOUR_TURN_HOST:3478"], "username": "user", "credential": "pass"},
    ]
})

# ======================
# UTIL: 가중치 자동 다운로드
# ======================
def download_if_missing(path: str, url: str, chunk: int = 1 << 14):
    if not url or os.path.exists(path):
        return
    try:
        print(f"[weights] downloading {path} from {url} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for c in r.iter_content(chunk_size=chunk):
                    if c:
                        f.write(c)
        print(f"[weights] saved -> {path}")
    except Exception as e:
        print(f"[weights] warn: could not fetch {url} ({e}). Place '{path}' manually.")

download_if_missing(DETECT_WEIGHTS, WEIGHTS_URL_DET)
download_if_missing(CLASSIFY_WEIGHTS, WEIGHTS_URL_CLS)

# ======================
# GLOBAL (room 프레임 저장)
# ======================
ROOM_FRAMES: Dict[str, Tuple[np.ndarray, float]] = {}
ROOM_LOCK = threading.Lock()

# ======================
# MODEL LOAD
# ======================
@st.cache_resource(show_spinner=False)
def load_models():
    det = YOLO(DETECT_WEIGHTS)        # 1단계: 탐지
    cls = YOLO(CLASSIFY_WEIGHTS)      # 2단계: ✅ 네 튜닝 분류기
    return det, cls

det_model, cls_model = load_models()

# ======================
# INFERENCE: 탐지 → 분류 → 표기
# ======================
def two_stage_annotate(frame_bgr: np.ndarray) -> np.ndarray:
    vis = frame_bgr.copy()
    det_res = det_model.predict(source=frame_bgr, imgsz=640, conf=CONF_THRES, verbose=False)

    crops, boxes = [], []
    for r in det_res:
        names = r.names or {}
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_id = int(b.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_bgr.shape[1]-1, x2), min(frame_bgr.shape[0]-1, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops.append(((x1, y1, x2, y2), crop))
                        boxes.append((x1, y1, x2, y2))

    labels = {}
    if crops:
        batch = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for _, c in crops]
        cls_res = cls_model.predict(source=batch, imgsz=224, verbose=False)
        for (xyxy, _), cr in zip(crops, cls_res):
            top1 = int(cr.probs.top1)
            name = cr.names.get(top1, str(top1))
            confp = float(cr.probs.top1conf)
            labels[xyxy] = f"{name} ({confp:.2f})"

    for (x1, y1, x2, y2) in boxes:
        label = labels.get((x1, y1, x2, y2), "vehicle")
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return vis

# ======================
# UI
# ======================
st.set_page_config(page_title="YOLO Two-Stage (iPhone→Laptop, WebRTC)", layout="wide")
st.title("🚗 YOLO Two-Stage: iPhone Camera → Server → Laptop (WebRTC)")

missing = [p for p in [DETECT_WEIGHTS, CLASSIFY_WEIGHTS] if not Path(p).exists()]
if missing:
    st.error(f"Missing weights: {missing}\n- Set auto-download URLs or place files in project root.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    st.write("• iPhone Safari/Chrome에서 이 페이지 접속 → Broadcast 탭 → 카메라 허용")
    st.write("• HTTPS 권장(필수), 안정성 위해 TURN 서버 추천")
    room_id = st.text_input("Room ID (iPhone & Laptop 동일하게)", value="demo")

tabs = st.tabs(["📱 Broadcast (iPhone)", "💻 View (Laptop)"])

# ---- iPhone: 방송 ----
with tabs[0]:
    st.subheader("📱 Broadcast")
    st.write("iPhone에서 이 탭을 열고 **카메라 허용** 후 시작하세요.")

    def publisher_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # 성능 필요 시 다운스케일: img = cv2.resize(img, (640, int(img.shape[0]*640/img.shape[1])))
        out = two_stage_annotate(img)
        with ROOM_LOCK:
            ROOM_FRAMES[room_id] = (out, time.time())
        return av.VideoFrame.from_ndarray(out, format="bgr24")

    webrtc_streamer(
        key=f"pub:{room_id}",
        mode=WebRtcMode.SENDRECV,  # iPhone 화면에도 주석 결과 표시
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=publisher_callback,
    )

# ---- 노트북: 시청 ----
with tabs[1]:
    st.subheader("💻 View")
    st.write("노트북에서는 같은 Room ID를 입력하고 아래 버튼으로 시청하세요.")
    start = st.button("뷰 시작")
    stop = st.button("뷰 정지")
    if "viewing" not in st.session_state:
        st.session_state.viewing = False
    if start: st.session_state.viewing = True
    if stop:  st.session_state.viewing = False

    canvas = st.empty()
    info = st.empty()

    if st.session_state.viewing:
        while st.session_state.viewing:
            frame: Optional[np.ndarray] = None
            ts = 0.0
            with ROOM_LOCK:
                if room_id in ROOM_FRAMES:
                    frame, ts = ROOM_FRAMES[room_id]
            if frame is None:
                info.warning("대기 중… iPhone 탭에서 Broadcast를 시작하고 카메라를 허용하세요.")
                time.sleep(0.3)
                continue
            canvas.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            info.caption(f"room_id = {room_id} | last update: {time.strftime('%H:%M:%S', time.localtime(ts))}")
            time.sleep(0.03)
