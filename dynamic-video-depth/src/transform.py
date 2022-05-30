import numpy as np
import os

input_dir = "./datafiles/davis_processed/frames_midas_robust_CVD_change_intrinsics_1/dog"
reference_dir = "./datafiles/davis_processed/frames_midas/dog"
output_dir = "./datafiles/davis_processed/frames_midas_robust_CVD_change_intrinsics_1_intrinsics_pos/dog"

os.makedirs(output_dir, exist_ok=True)

for i in range(60):
    a = np.load(os.path.join(input_dir, f"frame_{i:05d}.npz"))
    b = np.load(os.path.join(reference_dir, f"frame_{i:05d}.npz"))
    
    result = {}
    for key in a.files:
        result[key] = a[key]
    
    result["intrinsics"] = b["intrinsics"]
    result["pose_c2w"] = b["pose_c2w"]
    # result["depth_mvs"] = b["depth_mvs"]
    # result["depth_pred"] = b["depth_pred"]
    # result["motion_seg"] = b["motion_seg"]
    
    np.savez(os.path.join(output_dir, f"frame_{i:05d}.npz"), **result)