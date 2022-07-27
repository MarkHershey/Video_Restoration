import os
from pathlib import Path
from typing import Union

import cv2
import tqdm


def downsize_image_pil(img_path, out_path, downscale_factor: float = 0.25):
    from PIL import Image, ImageOps

    assert Path(img_path).is_file()

    img = Image.open(img_path)
    # get the size of the image
    width, height = img.size
    # get the new size
    new_width = int(width * downscale_factor)
    new_height = int(height * downscale_factor)
    # resize the image
    img = ImageOps.fit(img, (new_width, new_height))
    # save the image
    img.save(out_path)
    return out_path


def downsize_image_cv2(
    img_path: Union[Path, str],
    out_path: Union[Path, str],
    downscale_factor: float = 0.25,
) -> None:
    assert Path(img_path).is_file()
    assert not Path(out_path).exists()

    img = cv2.imread(str(img_path))
    # get size of image
    height, width, channels = img.shape
    # print("Original size: {}x{}".format(width, height))

    # get new size
    new_height = int(height * downscale_factor)
    new_width = int(width * downscale_factor)
    # print("New size: {}x{}".format(new_width, new_height))

    img = cv2.resize(img, (new_width, new_height))
    cv2.imwrite(str(out_path), img)
    return None


def batch_downsize_images(
    in_dir: Union[Path, str],
    out_dir: Union[Path, str],
    downscale_factor: float = 0.25,
):
    in_dir = Path(in_dir)
    assert in_dir.is_dir()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_name in tqdm.tqdm(os.listdir(in_dir)):
        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = in_dir / img_name
        out_path = out_dir / img_name
        downsize_image_cv2(img_path, out_path, downscale_factor)


if __name__ == "__main__":
    batch_downsize_images("./", "./out")
