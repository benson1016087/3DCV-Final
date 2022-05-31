from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

from resize import resize


def parse():
    parser = ArgumentParser()
    parser.add_argument("input_dir_1", type=Path)
    parser.add_argument("input_dir_2", type=Path)

    return parser.parse_args()

def matrics(preds, gts):
    preds = np.array(preds).reshape(len(preds), -1)
    gts = np.array(gts).reshape(len(gts), -1)
    print(gts.min())
    
    RMSE = np.sqrt(((preds - gts) ** 2).mean(axis=-1)).mean()
    LRMSE = np.sqrt(((np.log(preds) - np.log(gts)) ** 2).mean(axis=-1)).mean()
    
    return {"RMSE": RMSE, "Log RMSE": LRMSE}

if __name__ == "__main__":
    args = parse()

    n = len(list(args.input_dir_2.iterdir()))
    preds, gts = [], []
    for i in range(n):
        pred_file = np.load(f"{args.input_dir_1}/batch{i:04d}.npz")
        gt_file = np.load(f"{args.input_dir_2}/frame_{i:05d}.npz")
        
        pred_depth = pred_file["depth"]
        gt_depth = resize(gt_file["depth"][..., None], (160, 384))

        preds.append(pred_depth)
        gts.append(gt_depth)
    
    print(matrics(preds, gts))
        