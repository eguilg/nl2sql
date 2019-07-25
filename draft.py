# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# logit = torch.tensor([
# 						[[5,-1], [3,4], [4,5]],
# 						[[3,1], [1,2], [2,5]],
# 					], dtype=torch.float)
# label = torch.tensor([
# 	[0, 0, -1],
# 	[0, 1, 1]
# ], dtype=torch.long)
# ce = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
# ce = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
# loss1 = F.cross_entropy(logit.transpose(1, 2), label, ignore_index=-1, reduction='mean')
# loss2 = F.cross_entropy(logit.transpose(1, 2), label, ignore_index=-1, reduction='none').mean()
# print(loss1*5, loss2*6)
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sqlnet.utils import *
from sqlnet.model.sqlbert import SQLBert, BertAdam, BertTokenizer
BERT_DIR = '/home/zydq/.torch/models/bert'
BERT_TOKENNIZER_PATH = '/home/zydq/.torch/models/bert/tokenizer/8a0c070123c1f794c42a29c6904beb7c1b8715741e235bee04aca2c7636fc83f.9b42061518a39ca00b8b52059fd2bede8daa613f8a8671500e518a8c29de8c00'
BERT_PRETRAINED_PATH = '/home/zydq/.torch/models/bert/pretrained/42d4a64dda3243ffeca7ec268d5544122e67d9d06b971608796b483925716512.02ac7d664cff08d793eb00d6aac1d04368a1322435e5fe0a27c70b0b3a85327f'
BERT_CHINESE_WWM = '/home/zydq/.torch/models/bert/chinese-bert_chinese_wwm_pytorch/'


train_sql, train_table, train_db, dev_sql, dev_table, dev_db = load_dataset(use_small=False)
tokenizer = BertTokenizer.from_pretrained(BERT_TOKENNIZER_PATH, do_lower_case=True)
perm = np.random.permutation(len(train_sql))
batch_size = 1
for st in range(len(train_sql) // batch_size + 1):
	if st * batch_size == len(perm):
		break
	ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
	st = st * batch_size
	q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, header_type = to_batch_seq(train_sql, train_table, perm, st, ed,
																					 tokenizer=tokenizer)
	raw_q = [train_sql[i]['question'] for i in perm[st:ed]]

