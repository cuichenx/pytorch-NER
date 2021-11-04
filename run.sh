#!/usr/bin/env bash

feature_extractor=${1}
batch_size=${2}
use_char=${3}
use_crf=${4}
split_name=${5}  # e.g. grouped_1
wandb_name=${6}
gpu_id=${7}
grouped_swap=${8}

echo "NAME IS ${wandb_name}"

if [ $use_char == 'mlp' ]; then
    use_char_str="--use_char --char_feature_extractor mlp"
elif [ $use_char == 'cnn' ]; then
    use_char_str="--use_char --char_feature_extractor cnn"
else
    use_char_str="--no_char"
fi

if [ $use_crf == 'True' ]; then
    use_crf_str="--use_crf"
else
    use_crf_str="--no_crf"
fi

if [ $grouped_swap == 'Clf' ]; then
    grouped_swap_str="--grouped_swap --do_attested_classification"
    echo "doing attested classification with grouped swap"
elif [ $grouped_swap == 'True' ]; then
    grouped_swap_str="--grouped_swap"
    echo "doing grouped swap"
else
    grouped_swap_str=""
    echo "not doing grouped swap"
fi

echo "Using GPU device ${gpu_id}"
export CUDA_VISIBLE_DEVICES=${gpu_id}

python main.py \
    --feature_extractor ${feature_extractor} \
    ${use_char_str} \
    ${use_crf_str} \
    --val_pos_ratio -1 \
    --data_path data/data.pth \
    --split_path data/split_${split_name}.pth \
    --batch_size ${batch_size} \
    --wandb_name ${wandb_name} \
    ${grouped_swap_str}