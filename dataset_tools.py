"""Grouped source splits and fresh dataset publication."""
import hashlib
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

EXTENSIONS={'.jpg','.jpeg','.png','.webp'}
def source_group(path):
    sidecar=path.with_suffix(path.suffix+'.source.json')
    return json.loads(sidecar.read_text())['source_sha256'] if sidecar.exists() else hashlib.sha256(path.read_bytes()).hexdigest()

def split_sources(src,val_ratio=.2,seed=42):
    src=Path(src)
    if not 0<val_ratio<1 or not src.is_dir(): raise ValueError('Source directory and val_ratio in (0,1) required')
    result,labels={},{}
    for cls in sorted(p for p in src.iterdir() if p.is_dir()):
        groups={}
        for path in sorted(cls.rglob('*')):
            if path.suffix.lower() not in EXTENSIONS or not path.is_file(): continue
            group=source_group(path)
            if group in labels and labels[group]!=cls.name: raise ValueError('Conflicting labels for one source')
            labels[group]=cls.name
            groups.setdefault(group,[]).append(path)
        if len(groups)<2: raise ValueError(f'{cls.name}: two independent sources required')
        keys=sorted(groups); random.Random(seed).shuffle(keys)
        count=max(1,min(len(keys)-1,round(len(keys)*val_ratio)))
        validation=set(keys[:count])
        for group,paths in groups.items():
            for path in paths: result[path]=(cls.name,'val' if group in validation else 'train',group)
    if not result: raise ValueError('No image classes found')
    return result

def stage_directory(dst):
    dst=Path(dst).resolve(); dst.parent.mkdir(parents=True,exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=dst.name+'.staging-',dir=dst.parent))

def publish(stage,dst,manifest):
    stage,dst=Path(stage),Path(dst).resolve()
    (stage/'manifest.json').write_text(json.dumps(manifest,indent=2))
    backup=None
    if dst.exists():
        backup=dst.with_name(dst.name+f'.previous-{time.time_ns()}'); dst.rename(backup)
    try: stage.rename(dst)
    except Exception:
        if backup is not None: backup.rename(dst)
        raise
    return backup

def clean_stage(stage):
    if Path(stage).exists(): shutil.rmtree(stage)
