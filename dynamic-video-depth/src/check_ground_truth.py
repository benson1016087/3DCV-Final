from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def parse():
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=Path)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    for file in args.input_dir.iterdir():
        data = np.load(file)

        print(file, data["depth"].max())
