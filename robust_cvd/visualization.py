from utils.visualization import visualize_depth_dir
from moviepy.editor import *
from argparse import ArgumentParser
from pathlib import Path


def get_args():
    args = ArgumentParser()
    args.add_argument("target_dir", type=Path)
    args.add_argument("--fps", type=int, default=10)

    return args.parse_args()


def main(args):
    depth_midas_dir = args.target_dir / "depth_midas2/depth"
    depth_vis_midas_dir = args.target_dir / "depth_vis_midas2"
    depth_vis_midas_dir.mkdir(exist_ok=True)
    visualize_depth_dir(depth_midas_dir, depth_vis_midas_dir)

    max_epoch = max(
        [0]
        + [
            int(n.stem.split("_")[1][1:])
            for n in list(Path(args.target_dir / "R0-100_hierarchical2_midas2/StD100.0_StR1.0_SmD0_SmR0.0").iterdir())
            if len(n.stem.split("_")) == 3
        ]
    )
    if max_epoch == 0:
        depth_result_dir = args.target_dir / f"R0-100_hierarchical2_midas2/StD100.0_StR1.0_SmD0_SmR0.0/depth_e0000/e0000_filtered/depth/"
    else:
        depth_result_dir = (
            args.target_dir
            / f"R0-100_hierarchical2_midas2/StD100.0_StR1.0_SmD0_SmR0.0/depth_e{max_epoch:04d}_opt/e{max_epoch:04d}_opt_filtered/depth/"
        )
    depth_vis_result_dir = args.target_dir / "depth_vis_result"
    depth_vis_result_dir.mkdir(exist_ok=True)
    visualize_depth_dir(depth_result_dir, depth_vis_result_dir)

    color_dir = args.target_dir / "color_down_png"
    clip_color = ImageSequenceClip(color_dir, fps=args.fps)
    clip_midas = ImageSequenceClip(depth_vis_midas_dir, fps=args.fps)
    clip_result = ImageSequenceClip(depth_vis_result_dir, fps=args.fps)

    clip_color = clip_color.set_duration(clip_result.duration)
    clip_midas = clip_midas.set_duration(clip_result.duration)

    clip_color.write_videofile(str(args.target_dir / "clip_color.mp4"), fps=args.fps)
    clip_midas.write_videofile(str(args.target_dir / "clip_midas.mp4"), fps=args.fps)
    clip_result.write_videofile(str(args.target_dir / "clip_result.mp4"), fps=args.fps)


if __name__ == "__main__":
    args = get_args()
    main(args)
