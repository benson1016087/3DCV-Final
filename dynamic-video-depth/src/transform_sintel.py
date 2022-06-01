from pathlib import Path
import numpy as np
import os
from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--use_gt_depth", action="store_true")

    return parser.parse_args()


args = parse()
track_name = args.input_dir.stem

for file_path in args.input_dir.iterdir():
    orig = np.load(file_path)
    gt = np.load(
        f"./robust_CVD/MPI-Sintel-npz/training/{track_name}/{file_path.stem}.npz"
    )

    result = {}
    for key in orig.files:
        result[key] = orig[key]

    lmat = np.identity(4)
    lmat[2:4] *= -1
    result["intrinsics"] = gt["intrinsic"]
    result["pose_c2w"] = np.linalg.inv(
        lmat @ np.vstack((gt["extrinsic"], np.array([0, 0, 0, 1])))
    )

    if args.use_gt_depth:
        result["depth_mvs"] = np.clip(gt["depth"], 0, 80)
        result["depth_pred"] = np.clip(gt["depth"], 0, 80)

    np.savez(file_path, **result)
