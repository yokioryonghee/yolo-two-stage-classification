import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from split_dataset import build_dataset
from augment_dataset import build_augmented

def sources(folder,n=10):
    cls=folder/'car'; cls.mkdir(parents=True,exist_ok=True)
    for i in range(n):
        array=np.full((40,40,3),i*7,dtype=np.uint8)
        Image.fromarray(array).save(cls/f'image{i}.png')
    return cls

def assert_disjoint(manifest):
    train={row['group'] for row in manifest if row['split']=='train'}
    val={row['group'] for row in manifest if row['split']=='val'}
    assert train and val and not train&val

def test_rerun_cannot_mix_old_split_files(tmp_path):
    src=tmp_path/'seed'; dst=tmp_path/'dataset'; sources(src)
    first,_=build_dataset(src,dst)
    sources(src,15)
    second,backup=build_dataset(src,dst)
    assert backup.is_dir()
    assert_disjoint(second)
    assert len(list(dst.glob('*/*/*')))==15
    assert {row['output'] for row in second}=={str(p.relative_to(dst)) for p in dst.glob('*/*/*')}

def test_crops_from_same_source_stay_together(tmp_path):
    cls=sources(tmp_path/'crops',4)
    for name in ['image0','image1']:
        (cls/(name+'.png.source.json')).write_text(json.dumps({'source_sha256':'same-source'}))
    manifest,_=build_dataset(tmp_path/'crops',tmp_path/'dataset')
    assert_disjoint(manifest)
    assert len({row['split'] for row in manifest if row['group']=='same-source'})==1

def test_train_only_augmentation_and_repeatable_manifest(tmp_path):
    sources(tmp_path/'seed',6)
    first,_=build_augmented(tmp_path/'seed',tmp_path/'dataset',imgsz=32,multiplier=2)
    second,_=build_augmented(tmp_path/'seed',tmp_path/'dataset',imgsz=32,multiplier=2)
    assert first==second
    assert_disjoint(first)
    val=[row for row in first if row['split']=='val']
    assert len(val)==len({row['source'] for row in val})

def test_invalid_sources_do_not_replace_old_dataset(tmp_path):
    sources(tmp_path/'seed',2); dst=tmp_path/'dataset'; build_dataset(tmp_path/'seed',dst)
    previous=(dst/'manifest.json').read_text()
    with pytest.raises(ValueError): build_dataset(tmp_path/'seed',dst,val_ratio=1)
    assert (dst/'manifest.json').read_text()==previous

def test_training_receives_device_and_exact_checkpoint(tmp_path,monkeypatch):
    import sys,types
    from train_classifier import train
    data=tmp_path/'dataset'; (data/'train').mkdir(parents=True); (data/'val').mkdir()
    run=tmp_path/'run'; (run/'weights').mkdir(parents=True); (run/'weights/best.pt').write_bytes(b'fake-test-only')
    calls=[]
    class FakeYOLO:
        def __init__(self,path): pass
        def train(self,**kwargs): calls.append(kwargs); return types.SimpleNamespace(save_dir=run)
    monkeypatch.setitem(sys.modules,'ultralytics',types.SimpleNamespace(YOLO=FakeYOLO))
    assert train(data,device='cpu')==(run/'weights/best.pt').resolve()
    assert calls[0]['device']=='cpu' and calls[0]['plots'] is False
