#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py --ca --gpu \
--output_dir ./submit/baseline_ep98_306_443.json　\
--logdir ./log/baseline1.log

