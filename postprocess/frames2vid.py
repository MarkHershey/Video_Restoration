import argparse
import glob
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import tqdm


def make_video(
    frames_dir: str,
    video_outpath: str,
    fps: int,
    fourcc: str,
    frame_fmt: str = "png",
    maker: str = "opencv",
):
    assert Path(frames_dir).is_dir()
    assert fps > 0
    assert isinstance(fourcc, str) and len(fourcc) == 4
    assert maker in ["opencv", "ffmpeg"]

    if Path(video_outpath).exists():
        print(video_outpath)
        proceed = input(f"Above video outpath already exist. Type 'f' to overwrite: ")
        if proceed == "f":
            Path(video_outpath).unlink()
        else:
            print("Abort")
            return

    fourcc = fourcc.lower()

    glob_pattern = f"{str(Path(frames_dir).resolve())}/*.{frame_fmt}"
    filenames = sorted(glob.glob(glob_pattern))
    num_frames = len(filenames)
    assert num_frames > 0

    img_sample = filenames[0]
    stem_name = Path(img_sample).stem
    name_len = len(stem_name)
    f_pattern = f"{str(Path(frames_dir).resolve())}/%0{name_len}d.{frame_fmt}"
    start_number = int(stem_name)

    # get image size from first frame
    # assume all frames have the same size
    img = cv2.imread(img_sample)
    height, width, channels = img.shape
    img_size = (height, width)

    print(f"Found {num_frames} frames in '{frames_dir}'")
    print(f"Frame size: {img_size}")
    print(f"Writing video to '{video_outpath}'")
    print(f"Using {maker} to make video...\n")

    print("=" * 80, "\n")

    if maker == "opencv":
        opencv_make_video(filenames, video_outpath, fps, fourcc, img_size)
    elif maker == "ffmpeg":
        ffmpeg_make_video(f_pattern, video_outpath, fps, start_number)

    return


def opencv_make_video(
    filenames: Iterable[str],
    video_outpath: str,
    fps: int,
    fourcc: str,
    img_size: tuple,
) -> None:
    out = cv2.VideoWriter(video_outpath, cv2.VideoWriter_fourcc(*fourcc), fps, img_size)

    # Create a video from the frames
    for filename in tqdm.tqdm(filenames):
        frame = cv2.imread(filename)
        out.write(frame)

    out.release()
    print("Video written to:", video_outpath)

    return


def ffmpeg_make_video(
    f_pattern: str, video_outpath: str, fps: int, start_number: int = 0
) -> None:
    # ffmpeg -framerate 24 -i img%03d.png output.mp4
    # cmd = f"ffmpeg -framerate {fps} -i {f_pattern} {video_outpath}"
    cmd = f"ffmpeg -framerate {fps} -start_number {start_number} -i {f_pattern} -c:v libx264 -r {fps} -pix_fmt yuv420p {video_outpath}"
    completed = subprocess.run(cmd, shell=True)
    if completed.returncode == 0:
        print("Video written to:", video_outpath)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="/Users/markhuang/Downloads/test/test-frames",
        help="Input directory containing frames",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="solo_new.mp4",
        help="Output video filepath",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="avc1",
        help="FourCC code for video codec",
    )
    parser.add_argument(
        "--frame_fmt",
        type=str,
        default="png",
        help="Frame format: 'png' or 'jpg'",
    )
    parser.add_argument(
        "--maker",
        type=str,
        default="opencv",
        help="Video maker: 'opencv' or 'ffmpeg'",
    )
    args = parser.parse_args()

    make_video(
        frames_dir=args.dir,
        video_outpath=args.out,
        fps=args.fps,
        fourcc=args.fourcc,
        frame_fmt=args.frame_fmt,
        maker=args.maker,
    )

    return


if __name__ == "__main__":
    main()
