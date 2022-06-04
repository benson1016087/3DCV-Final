# 3DCV-Final: Depth Estimation of Dynamic Objects
Group 19
R10922042 柯宏穎 R10922077 蔡秉辰 R10922095 黃秉迦

## Environment building
Go to the two directories and build the environment according to each `README.md`. 

## Prepare for Sintel dataset
```bash

```

## How to run the code
### Stage 1
```bash 
# Get the prediction of coarse depth, camera K, R, T, and mask
python main.py --video_file [VIDEO_PATH] --path [STAGE_1_OUTPUT_PATH] --save_intermediate_depth_streams_freq 1 --num_epochs 0 --post_filter --opt.adaptive_deformation_cost 10 --frame_range 0-100 --save_depth_visualization

# Convert to input of stage 2
python convert_google_input.py -p [STAGE_1_OUTPUT_PATH] -o google_input
```

If you want to run through whole Sintel validation set:
```bash
# Change the paths in `run_sintel.py` first
# Then run:
python run_sintel.py
```

### Stage 2
```bash

```
