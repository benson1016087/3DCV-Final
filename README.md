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
