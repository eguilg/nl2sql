import torch
from sqlnet.utils import *
from sqlnet.model.sqlbert import SQLBert, BertTokenizer
import argparse

BERT_DIR = '/home/zydq/.torch/models/bert'
BERT_TOKENNIZER_PATH = '/home/zydq/.torch/models/bert/tokenizer/8a0c070123c1f794c42a29c6904beb7c1b8715741e235bee04aca2c7636fc83f.9b42061518a39ca00b8b52059fd2bede8daa613f8a8671500e518a8c29de8c00'
BERT_PRETRAINED_PATH = '/home/zydq/.torch/models/bert/pretrained/42d4a64dda3243ffeca7ec268d5544122e67d9d06b971608796b483925716512.02ac7d664cff08d793eb00d6aac1d04368a1322435e5fe0a27c70b0b3a85327f'
BERT_CHINESE_WWM = '/home/zydq/.torch/models/bert/chinese-bert_chinese_wwm_pytorch/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Whether use gpu')
    parser.add_argument('--toy', action='store_true', help='Small batchsize for fast debugging.')
    parser.add_argument('--output_dir', type=str, default='', help='Output path of prediction result')
    args = parser.parse_args()

    n_word=300
    if args.toy:
        use_small=True
        gpu=args.gpu
        batch_size=8
    else:
        use_small=False
        gpu=args.gpu
        batch_size=10

    dev_sql, dev_table, dev_db, test_sql, test_table, test_db = load_dataset(use_small=use_small, mode='test')

    tokenizer = BertTokenizer.from_pretrained(BERT_CHINESE_WWM, do_lower_case=True)
    model = SQLBert.from_pretrained(BERT_CHINESE_WWM)
    model_path = 'saved_model/best_bert_model'
    print ("Loading from %s" % model_path)
    model.load_state_dict(torch.load(model_path))
    print ("Loaded model from %s" % model_path)

    dev_acc = epoch_acc(model, batch_size, dev_sql, dev_table, dev_db, tokenizer=tokenizer)
    print ('Dev Logic Form Accuracy: %.3f, Execution Accuracy: %.3f' % (dev_acc[1], dev_acc[2]))

    print ("Start to predict test set")
    predict_test(model, batch_size, test_sql, test_table, args.output_dir, tokenizer=tokenizer)
    print ("Output path of prediction result is %s" % args.output_dir)
