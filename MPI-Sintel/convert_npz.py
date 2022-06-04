import cv2
from pathlib import Path
import sys
from argparse import ArgumentParser, Namespace
import os
import numpy as np
from tqdm import tqdm

sys.path.append("MPI-Sintel-depth-training-20150305/sdk/python")
from sintel_io import flow_read, depth_read, cam_read, disparity_read, segmentation_read

TASKS = ["training"]
CHANNELS = ["clean", "final"]
FRAME_FOLDER = "MPI-Sintel-complete"
DEPTH_FOLDER = "MPI-Sintel-depth-training-20150305"
SEGMENT_FOLDER = "MPI-Sintel-segmentation-training-20150219"
STEREO_FOLDER = "MPI-Sintel-stereo-training-20150305"


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    for task in TASKS:
        tmp_dir = args.input_dir / FRAME_FOLDER / task
        for video in tqdm(os.listdir(str(tmp_dir / "clean"))):
            output_dir = args.output_dir / task / video
            frame_num = len(os.listdir(tmp_dir / "clean" / video))
            output_dir.mkdir(parents=True)
            print(video)
            for i in range(frame_num):
                res = {}
                res["frame_clean"] = cv2.imread(
                    str(args.input_dir / FRAME_FOLDER / task / "clean" / video / f"frame_{i+1:04d}.png")
                )
                res["frame_final"] = cv2.imread(
                    str(args.input_dir / FRAME_FOLDER / task / "final" / video / f"frame_{i+1:04d}.png")
                )
                res["invalid"] = (
                    cv2.imread(str(args.input_dir / FRAME_FOLDER / task / "invalid" / video / f"frame_{i+1:04d}.png"))[
                        ..., 0
                    ]
                    / 255
                )
                res["depth"] = depth_read(
                    str(args.input_dir / DEPTH_FOLDER / task / "depth" / video / f"frame_{i+1:04d}.dpt")
                )
                res["intrinsic"], res["extrinsic"] = cam_read(
                    str(args.input_dir / DEPTH_FOLDER / task / "camdata_left" / video / f"frame_{i+1:04d}.cam")
                )
                res["disparity"] = disparity_read(
                    str(args.input_dir / STEREO_FOLDER / task / "disparities" / video / f"frame_{i+1:04d}.png")
                )
                res["segmentation"] = segmentation_read(
                    str(args.input_dir / SEGMENT_FOLDER / task / "segmentation" / video / f"frame_{i+1:04d}.png")
                )

                np.savez(output_dir / f"frame_{i:05d}.npz", **res)
