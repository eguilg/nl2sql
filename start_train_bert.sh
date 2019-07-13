#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_bert.py --gpu --bs 12 --restore