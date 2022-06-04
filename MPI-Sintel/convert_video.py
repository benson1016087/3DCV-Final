import cv2
from pathlib import Path
import sys
from argparse import ArgumentParser, Namespace


TASKS = ["training", "test"]
CHANNELS = ["clean", "final"]

def get_args()->Namespace:
    parser = ArgumentParser()

    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("-fps", "--frames_per_second", type=int, default=20)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    for task in TASKS:
        for channel in CHANNELS:
            tmp_dir = args.input_dir / task / channel
            for folder in tmp_dir.iterdir():
                frames = [cv2.imread(str(frame_path)) for frame_path in sorted(folder.iterdir())]
                height, width, depth = frames[0].shape

                output_dir = Path(str(folder).replace(str(args.input_dir), str(args.output_dir)))
                output_dir.mkdir(parents=True)
                writer = cv2.VideoWriter(str(output_dir / "video.mp4"), cv2.VideoWriter_fourcc("m", "p", "4", "v"), args.frames_per_second, (width,height))
                for frame in frames:
                    writer.write(frame)
                
                writer.release()