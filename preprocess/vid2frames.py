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
    base = Path("./")
    original = base / "original"
    frames = base / "frames"

    extract_frames_from_video(original / "Java.mp4", frames / "Java")
    extract_frames_from_video(original / "Solo_cropped.mp4", frames / "Solo_cropped")
    extract_frames_from_video(original / "Solo.mp4", frames / "Solo")


if __name__ == "__main__":
    main()
