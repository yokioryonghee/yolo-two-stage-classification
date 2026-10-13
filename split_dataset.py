import argparse
import hashlib
import shutil
from pathlib import Path
from dataset_tools import split_sources,stage_directory,publish,clean_stage

def build_dataset(src,dst,val_ratio=.2,seed=42):
    src,dst=Path(src).resolve(),Path(dst).resolve()
    if src==dst or src in dst.parents or dst in src.parents: raise ValueError('Separate source/output directories required')
    assignments=split_sources(src,val_ratio,seed)
    stage=stage_directory(dst); manifest=[]
    try:
        for path,(cls,split,group) in assignments.items():
            name=hashlib.sha256(str(path.relative_to(src)).encode()).hexdigest()+path.suffix.lower()
            target=stage/split/cls/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            manifest.append(dict(source=str(path.relative_to(src)),group=group,split=split,output=str(target.relative_to(stage))))
        return manifest,publish(stage,dst,manifest)
    finally: clean_stage(stage)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--src',default='data/crops'); p.add_argument('--dst',default='data/dataset')
    p.add_argument('--val_ratio',type=float,default=.2); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); manifest,backup=build_dataset(a.src,a.dst,a.val_ratio,a.seed)
    print(f'Created {len(manifest)} files; previous dataset: {backup}')
if __name__=='__main__': main()
