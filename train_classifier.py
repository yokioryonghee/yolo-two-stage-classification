"""Train with an explicit device; return this run's checkpoint."""
import argparse
from pathlib import Path

def train(data='data/dataset',device='cpu',imgsz=224,epochs=30,batch=32):
    from ultralytics import YOLO
    if not (Path(data)/'train').is_dir() or not (Path(data)/'val').is_dir(): raise ValueError('Prepare train/val first')
    result=YOLO('yolov8n-cls.pt').train(data=str(Path(data).resolve()),device=device,imgsz=imgsz,
        epochs=epochs,batch=batch,lr0=.01,patience=10,project='runs/classify',name='car_model',plots=False)
    best=Path(result.save_dir)/'weights/best.pt'
    if not best.is_file(): raise RuntimeError('No best.pt produced')
    return best.resolve()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/dataset'); p.add_argument('--device',default='cpu')
    p.add_argument('--imgsz',type=int,default=224); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch',type=int,default=32)
    a=p.parse_args(); print(train(a.data,a.device,a.imgsz,a.epochs,a.batch))
if __name__=='__main__': main()
