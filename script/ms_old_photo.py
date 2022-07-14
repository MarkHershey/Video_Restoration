# -*- coding: utf-8 -*-
"""Bringing Old Photo Back to Life.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NEm6AsybIiC5TwTU_4DqDkQO0nFRB-uA

#◢ Bringing Old Photos Back to Life

This is a reference implementation of our CVPR 2020 paper [1], which  revives an old photo to modern style. Should you be making use of our work, please cite our paper [1].

---


#◢ Verify Runtime Settings

**<font color='#FF000'> IMPORTANT </font>**

In the "Runtime" menu for the notebook window, select "Change runtime type." Ensure that the following are selected:
* Runtime Type = Python 3
* Hardware Accelerator = GPU

#◢ Git clone
"""

# !git clone https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life.git photo_restoration

"""#◢ Set up the environment

"""

# Commented out IPython magic to ensure Python compatibility.
# pull the syncBN repo
# %cd photo_restoration/Face_Enhancement/models/networks
# !git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# !cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
# %cd ../../../

# %cd Global/detection_models
# !git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# !cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
# %cd ../../

# download the landmark detection model
# %cd Face_Detection/
# !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# !bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
# %cd ../

# download the pretrained model
# %cd Face_Enhancement/
# !wget https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Face_Enhancement/checkpoints.zip
# !unzip checkpoints.zip
# %cd ../

# %cd Global/
# !wget https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Global/checkpoints.zip
# !unzip checkpoints.zip
# %cd ../

# ! pip install -r requirements.txt

"""#◢ Run the code

### Restore photos (normal mode)
"""

# Commented out IPython magic to ensure Python compatibility.
# !rm -rf /content/photo_restoration/output
# %cd /content/photo_restoration/
input_folder = "test_images/old"
output_folder = "output"

import os

basepath = os.getcwd()
input_path = os.path.join(basepath, input_folder)
output_path = os.path.join(basepath, output_folder)
os.mkdir(output_path)

# !python run.py --input_folder /content/photo_restoration/test_images/old --output_folder /content/photo_restoration/output/ --GPU 0

import io
import IPython.display
import numpy as np
import PIL.Image


def imshow(a, format="png", jpeg_fallback=True):
    a = np.asarray(a, dtype=np.uint8)
    data = io.BytesIO()
    PIL.Image.fromarray(a).save(data, format)
    im_data = data.getvalue()
    try:
        disp = IPython.display.display(IPython.display.Image(im_data))
    except IOError:
        if jpeg_fallback and format != "jpeg":
            print(
                (
                    'Warning: image was too large to display in format "{}"; '
                    "trying jpeg instead."
                ).format(format)
            )
            return imshow(a, format="jpeg")
        else:
            raise
    return disp


def make_grid(I1, I2, resize=True):
    I1 = np.asarray(I1)
    H, W = I1.shape[0], I1.shape[1]

    if I1.ndim >= 3:
        I2 = np.asarray(I2.resize((W, H)))
        I_combine = np.zeros((H, W * 2, 3))
        I_combine[:, :W, :] = I1[:, :, :3]
        I_combine[:, W:, :] = I2[:, :, :3]
    else:
        I2 = np.asarray(I2.resize((W, H)).convert("L"))
        I_combine = np.zeros((H, W * 2))
        I_combine[:, :W] = I1[:, :]
        I_combine[:, W:] = I2[:, :]
    I_combine = PIL.Image.fromarray(np.uint8(I_combine))

    W_base = 600
    if resize:
        ratio = W_base / (W * 2)
        H_new = int(H * ratio)
        I_combine = I_combine.resize((W_base, H_new), PIL.Image.LANCZOS)

    return I_combine


filenames = os.listdir(os.path.join(input_path))
filenames.sort()

for filename in filenames:
    print(filename)
    image_original = PIL.Image.open(os.path.join(input_path, filename))
    image_restore = PIL.Image.open(os.path.join(output_path, "final_output", filename))

    display(make_grid(image_original, image_restore))

"""### Restore the photos with scratches"""

# !rm -rf /content/photo_restoration/output/*
# !python run.py --input_folder /content/photo_restoration/test_images/old_w_scratch/ --output_folder /content/photo_restoration/output/ --GPU 0 --with_scratch

import os

input_folder = "test_images/old_w_scratch"
output_folder = "output"
input_path = os.path.join(basepath, input_folder)
output_path = os.path.join(basepath, output_folder)

filenames = os.listdir(os.path.join(input_path))
filenames.sort()

for filename in filenames:
    print(filename)
    if filename.startswith("."):
        continue

    image_original = PIL.Image.open(os.path.join(input_path, filename))
    if filename == "10103-2.jpg":
        filename = "10103-2.png"
    image_restore = PIL.Image.open(os.path.join(output_path, "final_output", filename))

    display(make_grid(image_original, image_restore))

"""#◢ Try it on your own photos!"""

from google.colab import files
import shutil

upload_path = os.path.join(basepath, "test_images", "upload")
upload_output_path = os.path.join(basepath, "upload_output")

if os.path.isdir(upload_output_path):
    shutil.rmtree(upload_output_path)

if os.path.isdir(upload_path):
    shutil.rmtree(upload_path)

os.mkdir(upload_output_path)
os.mkdir(upload_path)

uploaded = files.upload()
for filename in uploaded.keys():
    shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, filename))

"""Run the processing with:"""

# !python run.py --input_folder /content/photo_restoration/test_images/upload --output_folder /content/photo_restoration/upload_output --GPU 0 --with_scratch --HR

"""### Visualize

Now you have all your results under the folder `upload_output` and you can *manually* right click and download them.

Here we use the child photos of celebrities from https://www.boredpanda.com/childhood-celebrities-when-they-were-young-kids/?utm_source=google&utm_medium=organic&utm_campaign=organic 
"""

filenames_upload = os.listdir(os.path.join(upload_path))
filenames_upload.sort()

filenames_upload_output = os.listdir(os.path.join(upload_output_path, "final_output"))
filenames_upload_output.sort()

for filename, filename_output in zip(filenames_upload, filenames_upload_output):
    image_original = PIL.Image.open(os.path.join(upload_path, filename))
    image_restore = PIL.Image.open(
        os.path.join(upload_output_path, "final_output", filename_output)
    )

    display(make_grid(image_original, image_restore))
    print("")

"""## Download your results


"""

output_folder = os.path.join(upload_output_path, "final_output")
print(output_folder)
os.system(f"zip -r -j download.zip {output_folder}/*")
files.download("download.zip")
