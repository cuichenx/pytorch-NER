#!/usr/bin/env bash

feature_extractor=${1}
batch_size=${2}
use_char=${3}
use_crf=${4}
split_name=${5}  # e.g. grouped_1
gpu_id=${6}

echo "Using GPU device ${gpu_id}"
CUDA_VISIBLE_DEVICES=${gpu_id}

if [ $use_char == 'mlp' ]; then
    use_char_str="--use_char --char_feature_extractor mlp"
else if [ $use_char == 'cnn' ]; then
    use_char_str="--use_char --char_feature_extractor cnn"
else
    use_char_str="--no_char"
fi

if [ $use_crf == 'True' ]; then
    use_crf_str="--use_crf"
else
    use_crf_str="--no_crf"
fi


python main.py \
    --feature_extractor ${feature_extractor} \
    ${use_char_str} \
    ${use_crf_str} \
    --data_path data/data.pth \
    --split_path data/split_${split_name}.pth \
    --batch_size ${batch_size}
