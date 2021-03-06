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
unzip MPI-Sintel-complete.zip -d MPI-Sintel-complete

# depth data
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip
unzip MPI-Sintel-depth-training-20150305.zip -d MPI-Sintel-depth-training-20150305

# stereo data
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-stereo-training-20150305.zip
unzip MPI-Sintel-stereo-training-20150305.zip -d MPI-Sintel-stereo-training-20150305

# segmentation data
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-segmentation-training-20150219.zip
unzip MPI-Sintel-segmentation-training-20150219.zip -d MPI-Sintel-segmentation-training-20150219

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
python main.py --video_file [VIDEO_PATH] --path [STAGE_1_OUTPUT_PATH] \
--save_intermediate_depth_streams_freq 1 --num_epochs 0 --post_filter \
--opt.adaptive_deformation_cost 10 --frame_range 0-100 \
--save_depth_visualization

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
# After getting each frame data from stage 1,
# prepare_preprocess_data.py can be used to generate all
# preprocessed data, please change the path mentioned in 
# the code to STAGE_1_OUTPUT_PATH.
# Or just run  the following code to prepare a signle data

cd dynamic-video-depth
cp <target file> ./datafiles/davis_processed/frames_midas[PATH_SUFFIX]
python scripts/preprocess/davis/generate_flows.py --suffix PATH_SUFFIX
python scripts/preprocess/davis/generate_sequence_midas.py --suffix PATH_SUFFIX

# After running above command, you will get flow_pairs[PATH_SUFFIX]
# and sequences_select_pairs_midas[PATH_SUFFIX] in
# ./dynamic-video-depth/datafiles/davis_processed
# Then, you can start to train the model by the following command
# track id is the data class (i.e. 'dog' in frames_midas)
# backbone is the midas initial weight, default would be the weight
# provided by original author
# path_suffix is mentioned in above command, which is used to
# distinct with other data

bash ./experiments/davis/train_sequence.sh \
    <gpu num> \
    [--track_id TRACK_ID] \
    [--backbone MIDAS_INITIAL_WEIGHT] \
    [--path_suffix] PATH_SUFFIX

# After training, testing would be automatically executed.
# The result would be stored in test_results/.
```
