from joblib import Parallel, delayed
from pathlib import Path
import os


def run(input_root, output_root, output_name="google_input"):
    print(input_root)
    robust_cvd_cmd = (
        f"python main.py --video_file {str(input_root / 'video.mp4')} --path {str(output_root)} "
        + "--save_intermediate_depth_streams_freq 1 --num_epochs 3 "
        + "--post_filter --opt.adaptive_deformation_cost 10 --frame_range 0-100 "
        + "--save_depth_visualization --save_checkpoints --batch_size 2 --pose_opt_freq 1"
    )
    os.system(robust_cvd_cmd)

    convert_cmd = f"python convert_google_input.py -p {str(output_root)} -o {output_name}"
    os.system(convert_cmd)


def mv(old_root, new_root):
    os.system(f"cp -r {old_root} {new_root}")
    print(old_root, new_root)


if __name__ == "__main__":
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    sintel_root = Path("data/MPI-Sintel-video/training/final")
    output_root = Path("output/Sintel/googckpt_3epoch")

    val_list = ["alley_1", "ambush_5", "bamboo_2", "bandage_1", "cave_2", "market_6", "shaman_2", "sleeping_1", "temple_2"]
    output_root.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=4, backend="multiprocessing")(delayed(run)(sintel_root / n, output_root / n) for n in val_list)

    # old_root = Path("output/Sintel_all/googckpt_3epoch")
    # new_root = Path("output/Sintel/googckpt_3epoch")
    # output_root.mkdir(parents=True, exist_ok=True)
    # Parallel(n_jobs=4, backend="multiprocessing")(delayed(mv)(old_root / n, new_root) for n in val_list)
