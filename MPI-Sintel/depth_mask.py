from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.append("MPI-Sintel-depth-training-20150305/sdk/python")
from sintel_io import depth_read


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("depth_dir", type=Path)
    parser.add_argument("mask_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    frame_num = len(list(args.depth_dir.iterdir()))
    for i in range(frame_num):
        depth = depth_read(args.depth_dir / f"frame_{i+1:04d}.dpt")
        mask = cv2.imread(str(args.mask_dir / f"frame_{i+1:04d}.png"))[..., 0]
        masked_depth = np.where(mask == 0, depth, -1)
        print(depth.max(), masked_depth.max())
