# 3DCV-Final: Depth Estimation of Dynamic Objects
Group 19
R10922042 柯宏穎 R10922077 蔡秉辰 R10922095 黃秉迦

## Environment building
Go to the two directories and build the environment according to each `README.md`. 

## Prepare for Sintel dataset
```bash
cd MPI-Sintel

# basic data
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
unzip http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip -d MPI-Sintel-complete

# depth data
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip
unzip http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip -d MPI-Sintel-depth-training-20150305

# stereo data
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-stereo-training-20150305.zip
unzip http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-stereo-training-20150305.zip -d MPI-Sintel-stereo-training-20150305

# segmentation data
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-segmentation-training-20150219.zip
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-segmentation-training-20150219.zip -d MPI-Sintel-segmentation-training-20150219

# generate video data
python convert_video.py MPI-Sintel-complete/ MPI-Sintel-video

# generate extrinsic data
python convert_extrinsics.py MPI-Sintel-depth-training-20150305/training/camdata_left/ MPI-Sintel-extinsics

# generate npz data
python convert_npz.py . MPI-Sintel-npz

cd ..
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
