from pathlib import Path
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
from argparse import ArgumentParser


def get_args():
    args = ArgumentParser()
    args.add_argument("--pred_root", "-p", type=Path)
    args.add_argument("--gt", type=Path, default="data/debug/shuffle_False_gap_01_sequence_00000.pt")

    return args.parse_args()


def calc_err(rotq, tvec, rotq_gt, tvec_gt):
    r_err = np.linalg.norm((R.from_quat(rotq_gt) * R.from_quat(rotq).inv()).as_rotvec())
    t_err = np.linalg.norm(tvec - tvec_gt)

    return r_err, t_err


def get_pred_pose(pred, fid):
    # intrinsic = pred["intrinsic"][fid].numpy()
    intrinsic = pred["intrinsic"].numpy().mean(0)
    intrinsic = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])
    extrinsic = pred["extrinsic"][fid].numpy()

    camera_pose = intrinsic @ extrinsic
    return R.from_matrix(camera_pose[:, :-1]).as_quat(), camera_pose[:, -1].reshape(1, 3)


def get_gt_pose(gt, fid):
    K = gt[f"K"].numpy().squeeze().T
    rot = gt[f"R_{fid}"].numpy().squeeze()
    t = gt[f"t_{fid}"].numpy().squeeze().reshape(3, 1)
    return R.from_matrix(K @ rot).as_quat(), (K @ t).reshape(1, 3)


if __name__ == "__main__":
    args = get_args()
    pred = {
        "intrinsic": torch.load(args.pred_root / "intrinsic.pt"),
        "extrinsic": torch.load(args.pred_root / "extrinsic.pt"),
    }
    gt = torch.load(args.gt)
    print(gt.keys())
    print(gt["K"].squeeze().T)
    print(gt["depth_pred_1"].squeeze().T)
    exit()
    for i in range(1, 3):
        R_pred, t_pred = get_pred_pose(pred, int(gt[f"fid_{i}"].item()))
        R_gt, t_gt = get_gt_pose(gt, i)

        print(calc_err(R_pred, t_pred, R_gt, t_gt))
