import os
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np


def run(track_name):
    for p in Path(
        f"./datafiles/davis_processed/frames_midas_sintel_no_mask/{track_name}"
    ).iterdir():
        a = np.load(p)

        result = {}
        for key in a.files:
            result[key] = a[key]

        result.pop("motion_seg")
        np.savez(p, **result)

    cmd = (
        f"python scripts/preprocess/davis/generate_flows.py --track_id {track_name} --suffix _sintel_no_mask"
        + f" && python scripts/preprocess/davis/generate_sequence_midas.py --track_id {track_name} --suffix _sintel_no_mask"
    )

    os.system(cmd)


os.system(
    "cp -r ./datafiles/davis_processed/frames_midas_sintel ./datafiles/davis_processed/frames_midas_sintel_no_mask"
)

track_name = [
    "alley_1",
    "ambush_5",
    "ambush_6",
    "bamboo_2",
    "bandage_1",
    "cave_2",
    "market_6",
    "shaman_2",
    "sleeping_1",
    "temple_2",
]
Parallel(n_jobs=5, backend="multiprocessing")(delayed(run)(t) for t in track_name)
