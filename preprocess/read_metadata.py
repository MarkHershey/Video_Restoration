#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import cv2


def get_fourcc(fourcc_str: str):
    fourcc_int = cv2.VideoWriter_fourcc(*fourcc_str)
    return fourcc_int


def hex_to_decimal(hex_str: str):
    return int(hex_str, 16)


def get_fourcc_str(cap: cv2.VideoCapture) -> str:
    # get fourcc from opencv
    fourcc_decimal: int = int(cap.get(cv2.CAP_PROP_FOURCC))
    # convert decimal to hex
    fourcc_hex: str = hex(fourcc_decimal)
    assert len(fourcc_hex) == 10, "Invalid fourcc"
    # remove 0x from hex
    fourcc_hex = fourcc_hex[2:]
    # convert hex to ascii
    fourcc_str: str = ""
    for i in range(0, len(fourcc_hex), 2):
        fourcc_str += chr(hex_to_decimal(fourcc_hex[i : i + 2]))
    return fourcc_str[::-1].upper()


def read_metadata(video_filepath: str):
    try:
        # get OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
        print(f"OpenCV version: {major_ver}.{minor_ver}.{subminor_ver}")

        # create a VideoCapture object and read from input file
        cap: cv2.VideoCapture = cv2.VideoCapture(video_filepath)

        # get video metadata
        # ref: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        metadata = dict(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            # fourcc=int(cap.get(cv2.CAP_PROP_FOURCC)),
            fourcc=get_fourcc_str(cap),
            frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT),
            brightness=cap.get(cv2.CAP_PROP_BRIGHTNESS),
            contrast=cap.get(cv2.CAP_PROP_CONTRAST),
            saturation=cap.get(cv2.CAP_PROP_SATURATION),
            hue=cap.get(cv2.CAP_PROP_HUE),
            gain=cap.get(cv2.CAP_PROP_GAIN),
            convert_rgb=cap.get(cv2.CAP_PROP_CONVERT_RGB),
        )
        print(json.dumps(metadata, indent=4))

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) != 2:
        print("Usage: read_metadata.py <video_filepath>")
        sys.exit(1)
    else:
        video_filepath = sys.argv[1]
        assert Path(video_filepath).exists(), "Video file does not exist"
        read_metadata(video_filepath)


if __name__ == "__main__":
    main()
