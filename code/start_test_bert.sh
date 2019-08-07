#!/usr/bin/bash
python test_bert.py  --gpu \
                     --batch_size 12 \
                     --data_dir ../data/ \
                     --bert_model_dir ../model/chinese-bert_chinese_wwm_pytorch/ \
                     --restore_model_path ../model/best_bert_model

