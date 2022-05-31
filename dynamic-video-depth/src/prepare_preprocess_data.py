from typing import List
from joblib import Parallel, delayed
from pathlib import Path
from argparse import ArgumentParser

import os


def parse():
    parser = ArgumentParser()
    # parser.add_argument("--track_id", nargs="*", required=True)

    return parser.parse_args()


def run(input_root, output_root, args):
    output_name = output_root / "frames_midas_sintel" / input_root.stem
    output_name.mkdir(mode=0o755, parents=True, exist_ok=True)

    print(output_name)
    move_cmd = f"cp {input_root}/google_input/* {output_name}"
    preprocess_cmd = (
        f"python src/resize.py --input_dir {output_name} --height 160 --width 384"
        f" && python scripts/preprocess/davis/generate_flows.py --track_id {input_root.stem} --suffix _sintel"
        + f" && python scripts/preprocess/davis/generate_sequence_midas.py --track_id {input_root.stem} --suffix _sintel"
    )

    print(move_cmd, preprocess_cmd)
    os.system(move_cmd)
    os.system(preprocess_cmd)


if __name__ == "__main__":
    # os.environ["MKL_THREADING_LAYER"] = "GNU"

    args = parse()

    sintel_root = Path("./robust_CVD/Sintel/googckpt_3epoch")
    output_root = Path("./datafiles/davis_processed")
    output_root.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=3, backend="multiprocessing")(
        delayed(run)(p, output_root, args) for p in sintel_root.iterdir()
    )
