from argparse import ArgumentParser, Namespace
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import numpy as np
from resize import resize


def parse() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input_dir_1", type=Path)
    parser.add_argument("input_dir_2", type=Path)
    parser.add_argument("--all", help="run result of all sintel", action="store_true")
    parser.add_argument("--per_frame_scaling", "-pfs", action="store_true")

    return parser.parse_args()


def get_metrics(preds, gts, valids=None, acc_thresholds=[1.25, 1.25**2, 1.25**3], per_frame_scaling=False):
    if valids is None:
        valids = np.ones_like(gts) > 0

    preds = np.array(preds).reshape(len(preds), -1)
    gts = np.array(gts).reshape(len(gts), -1)
    valids = np.array(valids).reshape(len(valids), -1)
    print(gts[valids].min(), gts[valids].max())
    print(preds[valids].min(), preds[valids].max())

    if per_frame_scaling:
        scales = [np.median(gt[valid] / pred[valid]) for pred, gt, valid in zip(preds, gts, valids)]
    else:
        scales = [np.median(gts[valids] / preds[valids]) for _ in range(len(preds))]
    # print(scales)

    RMSE, LRMSE, abs_rel, square_rel = [], [], [], []
    for pred, gt, valid, scale in zip(preds, gts, valids, scales):
        pred, gt = scale * pred[valid], gt[valid]
        RMSE.append(np.sqrt(((pred - gt) ** 2).mean()))
        LRMSE.append(np.sqrt(((np.log(pred) - np.log(gt)) ** 2).mean()))
        abs_rel.append((np.abs(pred - gt) / gt).mean())
        square_rel.append((np.square(pred - gt) / gt).mean())

    res = {
        "RMSE": np.mean(RMSE),
        "Log RMSE": np.mean(LRMSE),
        "Abs Rel": np.mean(abs_rel),
        "Sq Rel": np.mean(square_rel),
    }

    for acc_threshold in acc_thresholds:
        acc = []
        for pred, gt, valid, scale in zip(preds, gts, valids, scales):
            pred, gt = scale * pred[valid], gt[valid]
            acc.append((np.maximum(gt / (pred + 1e-6), pred / (gt + 1e-6)) < acc_threshold).mean())
        res[f"acc-{acc_threshold}"] = np.mean(acc)

    return res


def get_one_video_metrics(pred_dir, gt_dir, per_frame_scaling=False):
    depth_maximum = 80

    n = len(list(gt_dir.iterdir()))
    preds, gts, valids = [], [], []
    for i in range(n):
        pred_file = np.load(f"{pred_dir}/batch{i:04d}.npz")
        gt_file = np.load(f"{gt_dir}/frame_{i:05d}.npz")

        valid = resize(((gt_file["depth"] > depth_maximum) | (gt_file["invalid"] != 0))[..., None], (160, 384)) == 0
        gt_depth = resize(np.minimum(gt_file["depth"], depth_maximum)[..., None], (160, 384))
        pred_depth = np.minimum(pred_file["depth"], depth_maximum)

        preds.append(pred_depth)
        gts.append(gt_depth)
        valids.append(valid)

    return get_metrics(preds, gts, valids, per_frame_scaling=per_frame_scaling)


if __name__ == "__main__":
    args = parse()
    if not args.all:
        metrics = get_one_video_metrics(args.input_dir_1, args.input_dir_2, args.per_frame_scaling)
        print(metrics)
    else:
        val_list = [
            "alley_1",
            "ambush_5",
            "bamboo_2",
            "bandage_1",
            "cave_2",
            "market_6",
            "shaman_2",
            "sleeping_1",
            "temple_2",
        ]

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
