import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
from utils import image_io

sys.path.append("../3DCV-Final/dynamic-video-depth/src")
from evaluate import get_metrics, parse
from resize import resize


def load_pred_file(pred_dir, idx):
    fname = pred_dir / "R0-100_hierarchical2_midas2/StD100.0_StR1.0_SmD0_SmR0.0" / f"depth_e0002_opt/e0002_opt_filtered/depth/frame_{idx:06d}.raw"
    return image_io.load_raw_float32_image(fname)


def get_one_video_metrics(pred_dir, gt_dir, per_frame_scaling=False):
    depth_maximum = 80

    n = len(list(gt_dir.iterdir()))
    preds, gts, valids = [], [], []
    for i in range(n):
        pred_file = load_pred_file(pred_dir, i)
        gt_file = np.load(f"{gt_dir}/frame_{i:05d}.npz")

        valid = resize(((gt_file["depth"] > depth_maximum) | (gt_file["invalid"] != 0))[..., None], (160, 384)) == 0
        gt_depth = resize(np.minimum(gt_file["depth"], depth_maximum)[..., None], (160, 384))
        pred_depth = np.minimum(pred_file, depth_maximum)

        preds.append(pred_depth)
        gts.append(gt_depth)
        valids.append(valid)

    return get_metrics(preds, gts, valids, per_frame_scaling=per_frame_scaling)


if __name__ == "__main__":
    args = parse()
    val_list = ["alley_1", "ambush_5", "bamboo_2", "bandage_1", "cave_2", "market_6", "shaman_2", "sleeping_1", "temple_2"]

    res = [track_id for track_id in val_list if track_id in str(args.input_dir_1)]
    assert len(res) == 1
    replaced = res[0]
    metrics_rec = defaultdict(list)
    for track_id in val_list:
        input_dir = Path(str(args.input_dir_1).replace(replaced, track_id))
        gt_dir = Path(str(args.input_dir_2).replace(replaced, track_id))
        # print(input_dir,gt_dir)
        ret = get_one_video_metrics(input_dir, gt_dir, args.per_frame_scaling)
        metrics_rec = {k: metrics_rec[k] + [ret[k]] for k in ret}
        print(track_id, ret)
    metrics_rec = {k: sum(v) / len(v) for k, v in metrics_rec.items()}
    print(f"Among all tracks: {metrics_rec}")
