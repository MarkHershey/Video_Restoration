import subprocess

from pathlib import Path


def run_cmd(cmd: str):
    print(cmd)
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return


BASE = Path("./photo_restoration")
assert BASE.is_dir()

IN_DIR = Path("/home/markhuang/code/Video_Restoration/preprocess/frames")
OUT_DIR = Path("/home/markhuang/Data/solo_out_1")
assert IN_DIR.is_dir()
assert OUT_DIR.is_dir()

cmd = f"python photo_restoration/run.py --input_folder {IN_DIR} --output_folder {OUT_DIR} --GPU 0 --with_scratch --HR"

run_cmd(cmd)
