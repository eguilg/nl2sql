#!/usr/bin/bash
python train_bert.py --gpu \
                     --batch_size 12 \
                     --lr 6e-6 \
                     --epoch 30 \
                     --data_dir ../data/ \
                     --bert_model_dir ../model/chinese-bert_chinese_wwm_pytorch/ \
                     --model_save_path ../model/saved_bert_model


