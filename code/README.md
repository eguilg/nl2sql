### Train
This command will train a new model with our default settings, model weights is saved at ```../model/saved_bert_model```.
```
cd code
sh start_train_bert.sh
```

You can also directly run the python script using your own arguments by:
```
cd code
python train_bert.py --gpu \
                     --batch_size 12 \
                     --lr 6e-6 \
                     --epoch 30 \
                     --data_dir ../data/ \
                     --bert_model_dir ../model/chinese-bert_chinese_wwm_pytorch/ \
                     --model_save_path ../model/your_model_name
``` 
### Test
This command will generate a submission file on the test set at ```../submit/result-xxxxxxxxxxxx.json``` with our best model at ```../model/best_bert_model```.

```
cd code
sh start_test_bert.sh
```
You can also directly run the python script using your own arguments by:
```
cd code 
python test_bert.py  --gpu \
                     --batch_size 12 \
                     --data_dir ../data/ \
                     --bert_model_dir ../model/chinese-bert_chinese_wwm_pytorch/ \
                     --restore_model_path ../model/model_you_have_trained 
``` 