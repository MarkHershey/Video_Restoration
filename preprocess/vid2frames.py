import argparse
from pathlib import Path

import cv2
import tqdm


def extract_frames_from_video(video_filepath: str, output_dir: str = "./frames"):
    inpath = Path(video_filepath)
    assert inpath.exists(), "Video file does not exist"
    outpath = Path(output_dir)
    outpath.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(inpath))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {length}")
    estimate_space_required(cap)
    zfill_val = len(str(length))

    for i in tqdm.tqdm(range(length)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(outpath / f"{i:0{zfill_val}}.jpg"), frame)

    cap.release()
    cv2.destroyAllWindows()


def estimate_space_required(cap):
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    test_file = "test_frame.jpg"
    if ret:
        cv2.imwrite(test_file, frame)
        test_file_size = Path(test_file).stat().st_size
        print(
            f"Estimated space required: {test_file_size * length / 1024 / 1024 / 1034:.2f} GB"
        )
        # delete test file
        Path(test_file).unlink()

    return


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("v_path", help="Path to video file")
    parser.add_argument(
        "--out_dir",
        default="./frames",
        help="Path to output directory",
    )
    args = parser.parse_args()
    extract_frames_from_video(args.v_path, args.out_dir)


if __name__ == "__main__":
    main()
