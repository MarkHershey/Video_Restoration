import argparse
import os
from pathlib import Path
from typing import *

import cv2
import imgaug.augmenters as iaa
import imgaug.parameters as iap
import numpy as np
import tqdm


def read_image(
    filename: str,
    colored: bool = True,
) -> np.ndarray:
    assert Path(filename).is_file()
    img = cv2.imread(
        str(filename),
        cv2.IMREAD_COLOR if colored else cv2.IMREAD_GRAYSCALE,
    )
    if colored:
        # convert to single channel grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # assert num of dimensions is 2
    assert img.ndim == 2

    # downscale by 0.25
    downscale_factor = 0.25
    img = cv2.resize(img, None, fx=downscale_factor, fy=downscale_factor)

    img = np.expand_dims(img, axis=2).astype(np.uint8)

    # get size of image
    height, width, channels = img.shape
    assert channels == 1

    # print(f"Image shape: {img.shape}")
    return img


def load_images(img_paths: List[str] = []):
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
    images = []
    for i in img_paths:
        images.append(read_image(i))

    return images


def process_batch(images: List[np.ndarray]) -> List[np.ndarray]:
    sometimes = lambda aug: iaa.Sometimes(0.35, aug)
    seq = iaa.Sequential(
        [
            iaa.OneOf(
                [
                    iaa.GaussianBlur(
                        (0, 3.0)
                    ),  # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(
                        k=(1, 7)
                    ),  # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(
                        k=(1, 11)
                    ),  # blur image using local medians with kernel sizes between 2 and 7
                    # iaa.imgcorruptlike.JpegCompression(severity=1),
                    # iaa.imgcorruptlike.ImpulseNoise(severity=1),
                ]
            ),
            # sometimes(iaa.Dropout((0.01, 0.05))),
            sometimes(iaa.Snowflakes(flake_size=(0.05, 0.3), speed=(0.001, 0.03))),
            iaa.OneOf(
                [
                    iaa.imgcorruptlike.Fog(severity=1),
                    iaa.imgcorruptlike.Contrast(severity=1),
                    iaa.BlendAlphaFrequencyNoise(
                        foreground=iaa.EdgeDetect(1.0),
                        # using only linear upscaling to scale the simplex noise masks to the final image sizes,
                        # i.e. This leads to rectangles with smooth edges.
                        upscale_method="linear",
                        sigmoid_thresh=iap.Normal(10.0, 5.0),
                        exponent=-2,
                    ),
                ]
            ),
        ],
        random_order=True,
    )
    images_aug = seq(images=images)
    return images_aug


def demo(source_img: str):
    images = load_images([source_img])
    seq = iaa.GaussianBlur((3, 3.0))
    images_1 = seq(images=images)
    save_batched_images(images_1, ["1.jpg"])

    seq = iaa.AverageBlur(k=(7, 7))
    images_2 = seq(images=images)
    save_batched_images(images_2, ["2.jpg"])

    seq = iaa.MedianBlur(k=(7, 11))
    images_3 = seq(images=images)
    save_batched_images(images_3, ["3.jpg"])

    seq = iaa.Dropout((0.01, 0.05))
    images_4 = seq(images=images)
    save_batched_images(images_4, ["4.jpg"])

    seq = iaa.Snowflakes(flake_size=(0.05, 0.3), speed=(0.001, 0.03))
    images_5 = seq(images=images)
    save_batched_images(images_5, ["5.jpg"])

    seq = iaa.imgcorruptlike.Fog(severity=1)
    images_6 = seq(images=images)
    save_batched_images(images_6, ["6.jpg"])

    seq = iaa.imgcorruptlike.Contrast(severity=1)
    images_7 = seq(images=images)
    save_batched_images(images_7, ["7.jpg"])

    seq = iaa.BlendAlphaFrequencyNoise(
        foreground=iaa.EdgeDetect(1.0),
        upscale_method="linear",
        sigmoid_thresh=iap.Normal(10.0, 5.0),
        exponent=-2,
    )
    images_8 = seq(images=images)
    save_batched_images(images_8, ["8.jpg"])


def save_batched_images(images: List[np.ndarray], out_paths: List[str] = []):
    assert len(images) == len(out_paths)
    for i in range(len(images)):
        cv2.imwrite(out_paths[i], images[i])


def batch_retro_images(
    in_dir: Union[Path, str],
    out_dir: Union[Path, str],
    batch_size: int = 128,
):
    in_dir = Path(in_dir)
    assert in_dir.is_dir()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = []
    out_paths = []

    for img_name in os.listdir(in_dir):
        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = in_dir / img_name
        out_path = out_dir / img_name
        img_paths.append(str(img_path))
        out_paths.append(str(out_path))

    for i in tqdm.tqdm(range(0, len(img_paths), batch_size)):
        batch_img_paths = img_paths[i : i + batch_size]
        batch_out_paths = out_paths[i : i + batch_size]

        images = load_images(batch_img_paths)
        images = process_batch(images)
        save_batched_images(images, batch_out_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        default="data/raw/original",
        help="Path to directory containing images to process",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/retro_style",
        help="Path to directory to save processed images",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of images to process in a batch",
    )
    args = parser.parse_args()
    batch_retro_images(args.in_dir, args.out_dir, args.batch_size)


if __name__ == "__main__":
    main()
    # demo("samples/grayscale.png")
