from argparse import ArgumentParser, Namespace
from pathlib import Path

from glob import glob
import cv2
import numpy as np


def parse():
    parser = ArgumentParser()

    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    args.output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    batch_data = glob(str(args.input_dir / "batch*.npz"))

    for i in range(len(batch_data)):
        data = np.load(batch_data[i])
        sf = data["sf_1_2"].squeeze()

        h = (np.arctan(sf[2] / sf[0]) + np.pi / 2) * 180 / np.pi * 2
        s = cv2.normalize(
            np.linalg.norm(sf, axis=0), None, 0, 5, cv2.NORM_MINMAX
        ).astype(np.uint8)
        v = np.ones_like(s) * 255

        cv2.imwrite(
            str(args.output_dir / f"frame_{i:05d}.png"),
            cv2.cvtColor(np.dstack((h, s, v)), cv2.COLOR_HSV2BGR),
        )
