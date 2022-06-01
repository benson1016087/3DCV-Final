from pathlib import Path
import numpy as np
from argparse import ArgumentParser
import torch


def parse():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)

    return parser.parse_args()


def resize(x, size):
    return (
        torch.nn.functional.interpolate(
            torch.tensor(x[None, ...], dtype=torch.float32).permute([0, 3, 1, 2]),
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        .permute([0, 2, 3, 1])
        .squeeze()
        .numpy()
    )


if __name__ == "__main__":
    args = parse()

    for input_file in args.input_dir.iterdir():
        a = np.load(input_file)

        result = {}
        for key in a.files:
            result[key] = a[key]

        scale_x, scale_y = (
            args.height / a["img"].shape[0],
            args.width / a["img"].shape[1],
        )
        size = [args.height, args.width]

        result["img"] = resize(a["img"], size)
        result["depth_mvs"] = resize(a["depth_mvs"][..., None], size)
        result["depth_pred"] = resize(a["depth_pred"][..., None], size)
        result["motion_seg"] = (resize(a["motion_seg"][..., None], size) > 0.5).astype(
            np.int64
        )
        result["intrinsics"][0] *= scale_x
        result["intrinsics"][1] *= scale_y

        np.savez(input_file, **result)
