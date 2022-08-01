import random
from pathlib import Path
from typing import Union

import cv2  # python-opencv
import numpy as np
from skimage.util import random_noise


def add_random_lines(img, min_num: int = 1, max_num: int = 5):
    num_lines = random.randint(min_num, max_num)
    # get width and height
    height, width = img.shape
    for _ in range(num_lines):
        # random starting point
        start_x = np.random.randint(0, width)
        start_y = np.random.randint(0, height)
        # random ending point
        end_x = np.random.randint(0, width)
        end_y = np.random.randint(0, height)
        # random line thickness
        line_thickness = random.randint(1, 3)
        # white color
        color = (255, 255, 255)
        cv2.line(
            img,
            (start_x, start_y),
            (end_x, end_y),
            color,
            thickness=line_thickness,
        )
    return img


def main(
    img_path: Union[str, Path],
    out_path: Union[str, Path],
    colored: bool = True,
    resize_factor: float = None,
    noise_type: str = "salt_and_pepper",
) -> None:
    assert Path(img_path).is_file()
    assert noise_type in ["gaussian", "salt_and_pepper"]
    # Load image
    img = cv2.imread(
        str(img_path),
        cv2.IMREAD_COLOR if colored else cv2.IMREAD_GRAYSCALE,
    )
    if colored:
        # convert to single channel grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get image dimensions
    height, width = img.shape
    print(f"Original Width: {width}, height: {height}")

    # resize image
    if resize_factor and 0 < resize_factor < 1:
        print(f"Resize factor: {resize_factor}")
        img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        height, width = img.shape
        print(f"New Width: {width}, height: {height}")

    # generate noise
    if noise_type == "gaussian":
        mean = 0
        var = 0.01
        sigma = var ** 0.5
        noise = np.random.normal(mean, sigma, (height, width))
        noise = noise.reshape(height, width) * 255
        img = img + noise
    elif noise_type == "salt_and_pepper":
        salt_vs_pepper = 0.5  # 0.0 < salt_vs_pepper < 1.0
        amount = 0.06  # 0.0 < amount < 1.0
        # Ref: https://scikit-image.org/docs/stable/api/skimage.util.html#random-noise
        img = random_noise(
            img,
            mode="s&p",
            amount=amount,
            salt_vs_pepper=salt_vs_pepper,
        )
        img = (255 * img).astype(np.uint8)

    # draw random lines on the image
    img = add_random_lines(img, min_num=2, max_num=5)
    # save noisy image
    cv2.imwrite(str(out_path), img)
    print("Image saved to: ", out_path)
    return


if __name__ == "__main__":
    main(
        img_path="../samples/good.png",
        out_path="../samples/noise.png",
        colored=True,
        resize_factor=0.5,
        noise_type="gaussian",
    )
