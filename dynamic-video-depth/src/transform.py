from pathlib import Path
import numpy as np
import os
from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--reference_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)

    return parser.parse_args()


args = parse()
args.output_dir.mkdir(parents=True, exist_ok=True)

for input_file, reference_file in zip(
    args.input_dir.iterdir(), args.reference_dir.iterdir()
):
    a = np.load(input_file)
    b = np.load(reference_file)

    result = {}
    for key in a.files:
        result[key] = a[key]

    # result["intrinsics"] = b["intrinsics"]
    # result["pose_c2w"] = b["pose_c2w"]
    # result["depth_mvs"] = b["depth_mvs"]
    # result["depth_pred"] = b["depth_pred"]
    # result["motion_seg"] = b["motion_seg"]

    result["depth_mvs"] = b["depth"]

    np.savez(args.output_dir / input_file.stem, **result)
