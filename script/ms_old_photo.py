import subprocess
from pathlib import Path


def run_cmd(cmd: str):
    print(cmd)
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return


HOME = Path("/home/markhuang")
ROOT = Path("/home/markhuang/code/Video_Restoration")
assert ROOT.is_dir()
BASE = ROOT / "script" / "photo_restoration"
assert BASE.is_dir()

IN_DIR = ROOT / "preprocess/frames"
IN_DIR = ROOT / "preprocess/test/test-frames"
assert IN_DIR.is_dir()
OUT_DIR = HOME / "Data/solo_out_1"
OUT_DIR = HOME / "code/Video_Restoration/preprocess/test/test-out"
assert OUT_DIR.is_dir()

cmd = f"python photo_restoration/run.py --input_folder {IN_DIR} --output_folder {OUT_DIR} --GPU 0 --with_scratch --HR"

run_cmd(cmd)
