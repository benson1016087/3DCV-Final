from os.path import join

import cv2
import numpy as np
from glob import glob

path = "/tmp3/laxingyang/3DCV/dynamic-video-depth/datafiles/davis_processed/frames_midas/dog"
file_names = sorted(glob(join(path, "*")))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("video.mov", fourcc, 10, (384, 192))

for file in file_names:
    img = cv2.cvtColor(np.load(file)["img"] * 255, cv2.COLOR_RGB2BGR).astype(np.uint8)
    video.write(img)

video.release()
