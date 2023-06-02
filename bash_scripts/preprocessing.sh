#!/bin/bash

export DATA_DIRECTORY=/home/pkr7098/python/Graduate.MaskCycleGAN-VC/vcc2018/vcc2018_training
export PREPROCESSED_DATA_DIRECTORY=/home/pkr7098/python/Graduate.MaskCycleGAN-VC/vcc_2018_preprocessed/vcc2018_training

python data_preprocessing/preprocess_vcc2018.py \
    --data_directory $DATA_DIRECTORY \
    --preprocessed_data_directory $PREPROCESSED_DATA_DIRECTORY \
    --speaker_id PKR_RESAMPLED dysarthria_resampled \
