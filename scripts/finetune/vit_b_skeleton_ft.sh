#!/usr/bin/env bash
set -x

JOB_NAME=$1
SPLIT_PATH=$2
PY_ARGS=${@:3}
OUTPUT_DIR="/anlab12/stuhome/zngz12/skeleton/exp_dir/${JOB_NAME}"
DATA_ROOT='/anlab12/stuhome/zngz12/skeleton/skeleton_dataset/reorganized_depth_noturn'
MODEL_PATH='/anlab12/stuhome/zngz12/skeleton/vit_b_k710_dl_from_giant.pth'

# Index->Real Map: 0->1, 1->2, 2->0 (do not support bf16)
export CUDA_VISIBLE_DEVICES=0,1

# .jpg for color, .png for depth

python -u -m torch.distributed.launch --nproc_per_node=2 \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set Skeleton \
        --nb_classes 3 \
        --fname_tmpl '{:05}.png' \
        --start_idx 0 \
        --data_path "${DATA_ROOT}/${SPLIT_PATH}_csv" \
        --data_root ${DATA_ROOT} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --num_sample 2 \
        --input_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --opt adamw \
        --lr 2e-3 \
        --num_workers 10 \
        --layer_decay 0.65 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --warmup_epochs 5 \
        --epochs 25 \
        --dist_eval \
        --test_num_segment 1 \
        --test_num_crop 2 \
        ${PY_ARGS} \
        --normed_depth \
        # --enable_deepspeed
