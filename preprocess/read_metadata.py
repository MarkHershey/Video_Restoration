#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import cv2


def read_metadata(video_filepath: str):
    try:
        # get OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
        print(f"OpenCV version: {major_ver}.{minor_ver}.{subminor_ver}")

        # create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(video_filepath)

        # get video metadata
        metadata = dict(
            width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            fps=cap.get(cv2.CAP_PROP_FPS),
            fourcc=cap.get(cv2.CAP_PROP_FOURCC),
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
