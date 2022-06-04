import cv2
from pathlib import Path
import sys
from argparse import ArgumentParser, Namespace
import os
import numpy as np
from tqdm import tqdm

sys.path.append("MPI-Sintel-depth-training-20150305/sdk/python")
from sintel_io import cam_read

TASKS = ["training"]
DEPTH_FOLDER = "MPI-Sintel-depth-training-20150305"


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    args.output_dir.mkdir(parents=True)
    for video in tqdm(os.listdir(str(args.input_dir))):
        frame_num = len(os.listdir(args.input_dir / video))
        print(video)
        res = np.array([cam_read(args.input_dir / video / f"frame_{i+1:04d}.cam")[1] for i in range(frame_num)])
        res = np.hstack((res, np.zeros((frame_num, 1, 4))))
        res[:, -1, -1] = 1
        print(res.shape)
        print(res[0])
        np.save(args.output_dir / f"{video}.npy", res)
