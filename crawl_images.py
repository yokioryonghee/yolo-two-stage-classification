#!/usr/bin/env python3
# crawl_images.py — DDG 이미지 크롤링 (레이트리밋 백오프, resume, no-new 조기종료)

import argparse, io, time, hashlib, random
from pathlib import Path

# ddgs 우선, 없으면 옛 패키지로 폴백
try:
    from ddgs import DDGS
except Exception:
    from duckduckgo_search import DDGS  # pragma: no cover

import requests
from PIL import Image

MIN_SIDE = 256
TIMEOUT = 15
UA_LIST = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
]

def safe_name(url: str) -> str:
    return f"{hashlib.sha1(url.encode()).hexdigest()}.jpg"

def download(url: str, out_path: Path) -> bool:
    try:
        headers = {"User-Agent": random.choice(UA_LIST)}
        r = requests.get(url, timeout=TIMEOUT, headers=headers)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        if min(im.size) < MIN_SIDE:
            return False
        im.save(out_path, format="JPEG", quality=92)
        return True
    except Exception:
        return False

def parse_queries(path: Path):
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    pairs = []
    for i, line in enumerate(lines, 1):
        parts = line.split("\t")
        if len(parts) < 2:
            print(f"[WARN] skip line {i} (need <class>\\t<query>): {line}")
            continue
        cls, q = parts[0].strip(), parts[1].strip()
        cnt = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else None
        pairs.append((cls, q, cnt))
    if not pairs:
        raise RuntimeError(f"No valid queries found in {path}")
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/queries.txt")
    ap.add_argument("--per_class", type=int, default=300, help="fallback count per line if not specified")
    ap.add_argument("--delay", type=float, default=0.4, help="sleep between downloads (sec)")
    ap.add_argument("--retries", type=int, default=6, help="rate-limit retries with backoff")
    ap.add_argument("--no_new_patience", type=int, default=5,
                    help="stop this class after this many consecutive batches with no new images")
    ap.add_argument("--out_root", default="data/raw")
    ap.add_argument("--region", default="wt-wt", help="ddg region (e.g., kr-kr, us-en)")
    ap.add_argument("--safesearch", default="off", choices=["off","moderate","strict"])
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for cls, q, cnt in parse_queries(Path(args.queries)):
        want = cnt or args.per_class
        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = {p.name for p in out_dir.glob("*.jpg")}
        got = len(existing)
        print(f"[crawl] {cls} :: {q} (target {want}, already {got})")

        attempt = 0              # rate-limit backoff counter
        empty_batches = 0        # consecutive batches with no new images

        while got < want:
            try:
                batch = min(100, want - got)  # 작은 배치로 반복
                with DDGS() as ddgs:
                    results = ddgs.images(q, max_results=batch,
                                          region=args.region, safesearch=args.safesearch)
                    n_ok = 0
                    for res in results:
                        url = res.get("image") or res.get("thumbnail")
                        if not url:
                            continue
                        name = safe_name(url)
                        if name in existing:
                            continue
                        if download(url, out_dir / name):
                            existing.add(name)
                            got += 1
                            n_ok += 1
                            if got % 25 == 0:
                                print(f"  + downloaded: {got}/{want}")
                        time.sleep(args.delay)

                if n_ok == 0:
                    empty_batches += 1
                    print(f"  ~ no new images in this batch ({empty_batches}/{args.no_new_patience})")
                    if empty_batches >= args.no_new_patience:
                        print("  -> no-new patience reached; moving to next class.")
                        break
                else:
                    empty_batches = 0  # reset on progress
                    attempt = 0        # reset rate-limit backoff on progress

            except Exception as e:
                msg = str(e)
                if "Ratelimit" in msg or "202" in msg:
                    wait = min(180, (2 ** attempt) * 15 + random.uniform(0, 3))  # 15,30,60,120,180...
                    attempt += 1
                    print(f"[RATE LIMIT] sleeping {wait:.1f}s (attempt {attempt}/{args.retries})")
                    time.sleep(wait)
                    if attempt > args.retries:
                        print("[ERROR] Too many rate-limit retries, moving to next class.")
                        break
                    continue
                else:
                    print(f"[ERROR] {e}")
                    break

        print(f"[done] {cls}: saved {got} images")

if __name__ == "__main__":
    main()
