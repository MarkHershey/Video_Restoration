from pathlib import Path
from typing import *

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap
import numpy as np


def read_image(
    filename: str,
    colored: bool = True,
    save_grayscale: bool = False,
) -> np.ndarray:
    assert Path(filename).is_file()
    img = cv2.imread(
        str(filename),
        cv2.IMREAD_COLOR if colored else cv2.IMREAD_GRAYSCALE,
    )
    if colored:
        # convert to single channel grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if save_grayscale:
            stem = Path(filename).stem
            cv2.imwrite("../samples/grayscale_" + str(stem) + ".png", img)

    img = np.expand_dims(img, axis=2).astype(np.uint8)

    # img = img.reshape(img.shape[0], img.shape[1], 1).astype(np.uint8)
    print(f"Image shape: {img.shape}")
    return img


def load_images():
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
    images = []
    paths = ["../samples/good.png"]
    for i in paths:
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
                        k=(2, 7)
                    ),  # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(
                        k=(3, 11)
                    ),  # blur image using local medians with kernel sizes between 2 and 7
                    iaa.imgcorruptlike.JpegCompression(severity=1),
                    iaa.imgcorruptlike.ImpulseNoise(severity=1),
                ]
            ),
            sometimes(iaa.Dropout((0.01, 0.05))),
            sometimes(iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.001, 0.03))),
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


def save_batched_images(images: List[np.ndarray]):
    for i in range(len(images)):
        cv2.imwrite(
            "../samples/retro_style_" + str(i) + ".png",
            images[i],
        )


def main():
    images = load_images()
    images = process_batch(images)
    save_batched_images(images)


if __name__ == "__main__":
    main()
