#!/bin/bash
python run.py \
    --frame_dir HMDB51 \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 51 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode 'train'