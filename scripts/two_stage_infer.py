"""Compatibility entry point for the maintained root script."""
import runpy
from pathlib import Path
if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'two_stage_infer.py'),run_name='__main__')
