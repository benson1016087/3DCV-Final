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

    return parser.parse_args()


def get_metrics(preds, gts, valids=None, acc_thresholds=[1.25, 1.25**2, 1.25**3]):
    if valids is None:
        valids = np.ones_like(gts) > 0

    preds = np.array(preds).reshape(len(preds), -1)
    gts = np.array(gts).reshape(len(gts), -1)
    valids = np.array(valids).reshape(len(valids), -1)
    print(gts.min(), gts.max())
    print(preds.min(), preds.max())

    RMSE, LRMSE, abs_rel, square_rel = [], [], [], []
    for pred, gt, valid in zip(preds, gts, valids):
        pred, gt = pred[valid], gt[valid]
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
        for pred, gt, valid in zip(preds, gts, valids):
            pred, gt = pred[valid], gt[valid]
            acc.append(
                (
                    np.maximum(gt / (pred + 1e-6), pred / (gt + 1e-6)) < acc_threshold
                ).mean()
            )
        res[f"acc-{acc_threshold}"] = np.mean(acc)

    return res


def get_one_video_metrics(pred_dir, gt_dir):
    clip_val = 80

    n = len(list(gt_dir.iterdir()))
    preds, gts, valids = [], [], []
    for i in range(n):
        pred_file = np.load(f"{pred_dir}/batch{i:04d}.npz")
        gt_file = np.load(f"{gt_dir}/frame_{i:05d}.npz")

        valid = resize(gt_file["invalid"][..., None], (160, 384)) == 0
        gt_depth = resize(np.minimum(gt_file["depth"], clip_val)[..., None], (160, 384))
        pred_depth = np.minimum(pred_file["depth"], clip_val)

        preds.append(pred_depth)
        gts.append(gt_depth)
        valids.append(valid)

    return get_metrics(preds, gts, valids)


if __name__ == "__main__":
    args = parse()
    if not args.all:
        metrics = get_one_video_metrics(args.input_dir_1, args.input_dir_2)
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
            ret = get_one_video_metrics(input_dir, gt_dir)
            metrics_rec = {k: metrics_rec[k] + [ret[k]] for k in ret}
            print(track_id, ret)
        metrics_rec = {k: sum(v) / len(v) for k, v in metrics_rec.items()}
        print(f"Among all tracks: {metrics_rec}")
