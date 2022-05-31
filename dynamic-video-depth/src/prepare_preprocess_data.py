from joblib import Parallel, delayed
from pathlib import Path
import os


def run(input_root, output_root):
    output_name = output_root / "frames_midas_sintel"
    output_name.mkdir(mode=0o755, parents=True, exist_ok=True)

    print(output_name)
    move_cmd = f"cp -r {input_root} {output_name / input_root.stem}"
    preprocess_cmd = (
        f"python src/transform.py --input_dir {output_name / input_root.stem} --reference_dir ./robust_CVD/MPI-Sintel-npz/training/{input_root.stem} --output_dir {output_name / input_root.stem}"
        f" && python scripts/preprocess/davis/generate_flows.py --track_id {input_root.stem} --suffix sintel"
        + f" && python scripts/preprocess/davis/generate_sequence_midas.py --track_id {input_root.stem} --suffix sintel"
    )

    print(move_cmd, preprocess_cmd)
    # os.system(move_cmd)
    # os.system(preprocess_cmd)


if __name__ == "__main__":
    # os.environ["MKL_THREADING_LAYER"] = "GNU"
    sintel_root = Path("./robust_CVD/Sintel/googckpt_3epoch")
    output_root = Path("./datafiles/davis_processed")
    output_root.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=3, backend="multiprocessing")(
        delayed(run)(p, output_root) for p in sintel_root.iterdir()
    )
