import os

for i in range(60):
    os.system(
        f"mv /tmp3/laxingyang/3DCV/dynamic-video-depth/datafiles/davis_processed/frames_midas_robust_CVD/dog/frame_{i:06d}.npz /tmp3/laxingyang/3DCV/dynamic-video-depth/datafiles/davis_processed/frames_midas_robust_CVD/dog/frame_{i:05d}.npz"
    )
