from pathlib import Path
import numpy as np
from argparse import ArgumentParser
import torch
import logging
import cv2

from utils import image_io


def get_args():
    args = ArgumentParser()
    args.add_argument("--pred_root", "-p", type=Path)
    args.add_argument("--output_name", "-o", type=str, default="google_input")

    return args.parse_args()


def get_pose_c2w(extrinsic):
    RT = np.zeros((4, 4))
    RT[:3, :] = extrinsic
    RT[-1, -1] = 1

    l_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    ret = np.linalg.inv(l_matrix @ np.linalg.inv(RT))
    return ret


def get_intrinsic(intrinsic: np.ndarray):
    if len(intrinsic.shape) > 1:
        intrinsic = intrinsic.mean(axis=0)

    return np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])


def get_motion_seg(video_root, frame_name):
    mask_img = cv2.imread(str(video_root / "dynamic_mask" / f"{frame_name}.png"))
    mask_img = (mask_img[..., 0] != 255).astype(int)

    return mask_img


def get_depth_pred(depth_root: Path, frame_stem: str):
    max_epoch = [
        int(n.stem.split("_")[1][1:]) for n in list(Path(depth_root / "StD100.0_StR1.0_SmD0_SmR0.0").iterdir()) if len(n.stem.split("_")) == 3
    ]
    if len(max_epoch) == 0:
        depth_result_dir = depth_root / f"StD100.0_StR1.0_SmD0_SmR0.0/depth_e0000/depth/"
    else:
        max_epoch = max(max_epoch)
        depth_result_dir = depth_root / f"StD100.0_StR1.0_SmD0_SmR0.0/depth_e{max_epoch:04d}_opt/depth/"
    robust_cvd_depth = image_io.load_raw_float32_image(depth_result_dir / f"{frame_stem}.raw")

    return 10000 / robust_cvd_depth


def read_img(img_path: Path):
    img = cv2.imread(str(img_path))
    img = img.astype(np.float) / 255

    return img[..., ::-1]


def main(predict_root, output_root):
    depth_root = predict_root / "R0-100_hierarchical2_midas2"
    logging.info(f"Saveing result to {output_root}")
    intrinsics = torch.load(depth_root / "intrinsic.pt").numpy()
    extrinsics = torch.load(depth_root / "extrinsic.pt").numpy()
    target_shape = (160, 384)

    assert intrinsics.shape[0] == extrinsics.shape[0]
    # intrinsic_avg = get_intrinsic(intrinsics)
    for i in range(intrinsics.shape[0]):
        frame_stem = f"frame_{i:06d}"
        output_frame_stem = f"frame_{i:05d}"

        img = read_img(predict_root / "color_full" / f"{frame_stem}.png")
        intrinsic_i = get_intrinsic(intrinsics[i])
        pose_c2w = get_pose_c2w(extrinsics[i])
        motion_seg = get_motion_seg(predict_root, frame_stem)
        depth_pred = get_depth_pred(depth_root, frame_stem)

        exit()
        np.savez(
            output_root / f"{output_frame_stem}.npz",
            img=img,
            pose_c2w=pose_c2w,
            depth_mvs=depth_pred,
            intrinsics=intrinsic_i,
            depth_pred=depth_pred,
            img_orig=img,
            motion_seg=motion_seg,
        )


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-6s %(message)s")

    pred_root = args.pred_root
    output_root = pred_root / args.output_name
    output_root.mkdir(exist_ok=True)

    logging.info(f"Output root of robust_cvd = {pred_root}")
    main(pred_root, output_root)
