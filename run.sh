#!/usr/bin/env bash
# PATIENT

gpu_id=${1}
feature_extractor=${2}
batch_size=${3}
use_char=${4}
use_crf=${5}
split_name=${6}  # e.g. grouped_1
wandb_name=${7}
grouped_swap=${8}
chardim=${9}
worddim=${10}
train_pos_ratio=${11}
char_after_cnn=${12}


echo "NAME IS ${wandb_name}"

if [ $use_char == 'phonemes' ]; then
    use_char_str="--use_char --char_feature_extractor mlp --what_char phonemes"
elif [ $use_char == 'tones' ]; then
    use_char_str="--use_char --char_feature_extractor mlp --what_char tones"
elif [ $use_char == 'letters' ]; then
    use_char_str="--use_char --char_feature_extractor cnn --what_char letters"
else
    use_char_str="--no_char"
fi

if [ $use_crf == 'True' ]; then
    use_crf_str="--use_crf"
else
    use_crf_str="--no_crf"
fi

if [ $char_after_cnn == 'True' ]; then
    char_after_cnn_str="--char_after_cnn"
else
    char_after_cnn_str=""
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
    --train_pos_ratio ${train_pos_ratio} \
    --data_path data/data.pth \
    --split_path data/split_${split_name}.pth \
    --batch_size ${batch_size} \
    --wandb_name ${wandb_name} \
    ${grouped_swap_str} \
    --char_hidden_dim ${chardim} --char_embedding_dim ${chardim} \
    --word_embed_dim ${worddim} \
    ${char_after_cnn_str}