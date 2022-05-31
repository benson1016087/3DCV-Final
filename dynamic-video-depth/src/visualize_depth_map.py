from os.path import join
from os import makedirs

import cv2
import numpy as np
from glob import glob

import colormaps

path = "/tmp3/laxingyang/3DCV/dynamic-video-depth/datafiles/davis_processed/frames_midas/dog"
output_path = "./dog_depth_map"

makedirs(output_path, exist_ok=True)
file_names = sorted(glob(join(path, "*")))

for file in file_names:
    img = np.load(file)["depth_pred"]
    img = (np.sqrt((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)

    img = ((cv2.applyColorMap(img, colormaps.cm_magma) / 255) ** 2.2) * 255
    cv2.imwrite(join(output_path, file.split("/")[-1].replace(".npz", ".png")), img)
