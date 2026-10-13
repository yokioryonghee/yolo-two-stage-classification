import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def detect_device():
    import torch
    if torch.cuda.is_available(): return 'cuda:0'
    if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['crawl','seed'],default='seed'); p.add_argument('--seed_dir',default='data/seed')
    p.add_argument('--queries',default='data/queries.txt'); p.add_argument('--per_class',type=int,default=300)
    p.add_argument('--crawl_delay',type=float,default=.6); p.add_argument('--retries',type=int,default=10)
    p.add_argument('--no_new_patience',type=int,default=5)
    p.add_argument('--det_weights',default='yolov8n.pt'); p.add_argument('--conf',type=float,default=.35)
    p.add_argument('--cls_conf',type=float,default=.6); p.add_argument('--cls_weights')
    p.add_argument('--device'); p.add_argument('--source',default='0'); p.add_argument('--skip_train',action='store_true')
    p.add_argument('--imgsz',type=int,default=224); p.add_argument('--val_ratio',type=float,default=.2)
    p.add_argument('--aug_multiplier',type=int,default=12); p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch',type=int,default=32); p.add_argument('--no_show',action='store_true')
    a=p.parse_args(); device=a.device or detect_device()
    if a.skip_train:
        if not a.cls_weights: p.error('--skip_train requires explicit --cls_weights')
        best=(ROOT/a.cls_weights).resolve()
        if not best.is_file(): p.error('Classifier checkpoint does not exist')
    else:
        if a.mode=='seed':
            run('augment_dataset.py','--src',a.seed_dir,'--dst','data/dataset','--imgsz',a.imgsz,'--val_ratio',a.val_ratio,'--multiplier',a.aug_multiplier)
        else:
            if not (ROOT/a.queries).is_file(): p.error('Provide a queries file with class-tab-query lines')
            run('crawl_images.py','--queries',a.queries,'--per_class',a.per_class,'--delay',a.crawl_delay,'--retries',a.retries,'--no_new_patience',a.no_new_patience)
            run('verify_images.py','--src','data/raw','--dst','data/clean')
            run('crop_from_raw.py','--src','data/clean','--dst','data/crops','--det_weights',a.det_weights,'--conf',a.conf,'--device',device)
            run('split_dataset.py','--src','data/crops','--dst','data/dataset','--val_ratio',a.val_ratio)
        from train_classifier import train
        import os
        previous=Path.cwd()
        try:
            os.chdir(ROOT); best=train('data/dataset',device,a.imgsz,a.epochs,a.batch)
        finally: os.chdir(previous)
    arguments=['--det_weights',a.det_weights,'--cls_weights',str(best),'--device',device,'--source',a.source,'--conf',a.conf,'--cls_conf',a.cls_conf]
    if not a.no_show: arguments.append('--show')
    run('two_stage_infer.py',*arguments)
if __name__=='__main__': main()
