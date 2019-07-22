import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertEncoder, BertAttention
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam, BertConfig


class SQLBert(BertPreTrainedModel):
	def __init__(self, config, hidden=150, gpu=True, dropout_prob=0.2, bert_cache_dir=None):
		super(SQLBert, self).__init__(config)
		self.OP_SQL_DIC = {0: ">", 1: "<", 2: "==", 3: "!="}
		self.AGG_DIC = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
		self.CONN_DIC = {0: "", 1: "and", 2: "or"}
		self.SQL_TOK = ['WHERE', 'AND', 'OR', '==', '>', '<', '!=']

		self.bert_cache_dir = bert_cache_dir

		self.bert = BertModel(config)
		self.apply(self.init_bert_weights)
		self.bert_hidden_size = self.config.hidden_size
		self.W_w_conn = nn.Linear(self.bert_hidden_size, 3)
		self.W_s_num = nn.Linear(self.bert_hidden_size, 5)
		self.W_w_num = nn.Linear(self.bert_hidden_size, 5)
		self.W_s_col = nn.Linear(self.bert_hidden_size, 1)
		self.W_s_agg = nn.Linear(self.bert_hidden_size, 6)
		self.W_w_col = nn.Linear(self.bert_hidden_size, 1)
		self.W_w_op = nn.Linear(self.bert_hidden_size, 4)

		self.W_q_s = nn.Linear(self.bert_hidden_size, hidden)
		self.W_col_s = nn.Linear(self.bert_hidden_size, hidden)
		self.W_q_e = nn.Linear(self.bert_hidden_size, hidden)
		self.W_col_e = nn.Linear(self.bert_hidden_size, hidden)
		self.W_w_s = nn.Linear(hidden, 1)
		self.W_w_e = nn.Linear(hidden, 1)

		self.dropout = nn.Dropout(dropout_prob)

		self.softmax = nn.Softmax(dim=-1)
		self.log_softmax = nn.LogSoftmax(dim=-1)

		self.kl_loss = nn.KLDivLoss(reduction='batchmean')

		self.gpu = gpu

		if gpu:
			self.cuda()

	def forward(self, inputs, return_logits=True):

		input_seq, q_mask, col_mask, col_index, token_type_ids, attention_mask = self.transform_inputs(inputs)
		out_seq, pooled_output = self.bert(input_seq, token_type_ids, attention_mask, output_all_encoded_layers=False)

		out_seq = self.dropout(out_seq)
		cls_raw = out_seq[:, 0]
		cls_emb = self.dropout(pooled_output)

		max_qlen = q_mask.shape[-1]
		max_col_num = col_mask.shape[-1]

		q_seq = out_seq[:, 1:1+max_qlen]

		# 这里将每一列的token各自聚合, 得到(None, max_col_len, 768)的 col_seq
		out_seq = out_seq.cumsum(dim=1)
		col_index = col_index.unsqueeze(-1).expand(-1,-1, out_seq.shape[-1])
		col_seq = out_seq.gather(dim=1, index=col_index[:, 0:1])  # 之前的和 (None, 1, 768)
		for i in range(1, col_index.shape[1]):
			next_sum = out_seq.gather(dim=1, index=col_index[:, i: i+1, :])  #  (None, 1, 768)
			interval = (col_index[:, i: i+1] - col_index[:, i-1: i]).float().clamp(1.0)
			col_seq[:, i-1: i, :] = (cls_raw.unsqueeze(1) + next_sum - col_seq[:, i-1: i, :]) / (interval + 1)
			if i != col_index.shape[1] - 1:
				col_seq = torch.cat([col_seq, next_sum], dim=1)

		where_conn_logit = self.W_w_conn(cls_emb)
		sel_num_logit = self.W_s_num(cls_emb)
		where_num_logit = self.W_w_num(cls_emb)
		sel_col_logit = self.W_s_col(col_seq).squeeze(-1)
		sel_agg_logit = self.W_s_agg(col_seq)
		where_col_logit = self.W_w_col(col_seq).squeeze(-1)
		where_op_logit = self.W_w_op(col_seq)

		q_col_s = F.leaky_relu(self.W_q_s(q_seq).unsqueeze(2) + self.W_col_s(col_seq).unsqueeze(1)) # (None, q, col, 768)
		q_col_e = F.leaky_relu(self.W_q_e(q_seq).unsqueeze(2) + self.W_col_e(col_seq).unsqueeze(1))  # (None, q, col, 768)
		where_start_logit = self.W_w_s(q_col_s).squeeze(-1)
		where_end_logit = self.W_w_e(q_col_e).squeeze(-1)

		where_conn_logit2, \
		sel_num_logit2, where_num_logit2, sel_col_logit2, \
		sel_agg_logit2, where_col_logit2, where_op_logit2, \
		where_start_logit2, where_end_logit2 = _get_logits(cls_emb, q_seq, col_seq, max_col_num)

		where_conn_logit = (where_conn_logit + where_conn_logit2) / 2
		sel_num_logit = (sel_num_logit + sel_num_logit2) / 2
		where_num_logit = (where_num_logit + where_num_logit2) / 2
		sel_col_logit = (sel_col_logit + sel_col_logit2) / 2
		sel_agg_logit = (sel_agg_logit + sel_agg_logit2) / 2
		where_col_logit = (where_col_logit + where_col_logit2) / 2
		where_op_logit = (where_op_logit + where_op_logit2) / 2
		where_start_logit = (where_start_logit + where_start_logit2) / 2
		where_end_logit = (where_end_logit + where_end_logit2) / 2


		# 联合概率
		# sel_agg_logit = sel_agg_logit + sel_col_logit[:, :, None]
		# where_op_logit = where_op_logit + where_col_logit[:, :, None]
		# where_start_logit = where_start_logit + where_op_logit.max(-1)[0][:, None, :]
		# where_end_logit = where_end_logit + where_start_logit.max(1)[0][:, None, :]


		# 处理mask, 因为masked_fill要求fill的位置mask为1,保留的位置mask为0
		q_mask, col_mask = q_mask.byte(), col_mask.byte()
		qcol_mask = q_mask.unsqueeze(2) & col_mask.unsqueeze(1)
		q_mask, col_mask, qcol_mask = ~q_mask, ~col_mask, ~qcol_mask
		# do mask
		sel_col_logit = sel_col_logit.masked_fill(col_mask, -1e5)
		sel_agg_logit = sel_agg_logit.masked_fill(col_mask.unsqueeze(-1).expand(-1, -1, 6), -1e5)
		where_col_logit = where_col_logit.masked_fill(col_mask, -1e5)
		where_op_logit = where_op_logit.masked_fill(col_mask.unsqueeze(-1).expand(-1, -1, 4), -1e5)
		where_start_logit = where_start_logit.masked_fill(qcol_mask, -1e5)
		where_end_logit = where_end_logit.masked_fill(qcol_mask, -1e5)

		return  where_conn_logit, \
				sel_num_logit, where_num_logit, sel_col_logit, \
				sel_agg_logit, where_col_logit, where_op_logit, \
				where_start_logit, where_end_logit



	def loss(self, logits, labels, q_lens, col_nums):

		where_conn_logit, \
		sel_num_logit, where_num_logit, sel_col_logit, \
		sel_agg_logit, where_col_logit, where_op_logit,\
		where_start_logit, where_end_logit = logits

		where_conn_label, sel_num_label, where_num_label, \
		sel_col_label, sel_agg_label, where_col_label, where_op_label, \
		where_start_label, where_end_label = self.transform_inputs(labels)

		# q_lens, col_nums = self.transform_inputs((q_lens, col_nums))
		# q_lens, col_nums = q_lens.float(), col_nums.float()

		# Evaluate the cond conn type
		where_conn_loss = F.cross_entropy(where_conn_logit, where_conn_label)
		sel_num_loss = F.cross_entropy(sel_num_logit, sel_num_label)
		where_num_loss = F.cross_entropy(where_num_logit, where_num_label)
		sel_agg_loss = F.cross_entropy(sel_agg_logit.transpose(1, 2), sel_agg_label, ignore_index=-1)
		where_op_loss = F.cross_entropy(where_op_logit.transpose(1, 2), where_op_label, ignore_index=-1)
		sel_col_loss = torch.abs(self.kl_loss(self.log_softmax(sel_col_logit), sel_col_label.float()))
		where_col_loss = torch.abs(self.kl_loss(self.log_softmax(where_col_logit), where_col_label.float()))
		where_start_loss = F.cross_entropy(where_start_logit, where_start_label, ignore_index=-1)
		where_end_loss = F.cross_entropy(where_end_logit, where_end_label, ignore_index=-1)

		loss = where_conn_loss + sel_num_loss + where_num_loss + sel_agg_loss \
			   + where_op_loss + sel_col_loss + where_col_loss + where_start_loss + where_end_loss

		return loss

	def transform_inputs(self, inputs):
		for x in inputs:
			if isinstance(x, (list, tuple)):
				x = np.array(x)
			if self.gpu:
				yield torch.from_numpy(np.array(x)).cuda()
			else:
				yield torch.from_numpy(np.array(x))

	def gen_query(self, logits, q, col, raw_q, reinforce=False, verbose=False):
		"""
		:param score:
		:param q: token-questions
		:param col: token-headers
		:param raw_q: original question sequence
		:return:
		"""
		where_conn_logit, \
		sel_num_logit, where_num_logit, sel_col_logit, \
		sel_agg_logit, where_col_logit, where_op_logit, \
		where_start_logit, where_end_logit = logits

		where_conn_logit = where_conn_logit.data.cpu().numpy()
		sel_num_logit = sel_num_logit.data.cpu().numpy()
		where_num_logit = where_num_logit.data.cpu().numpy()
		sel_col_logit = sel_col_logit.data.cpu().numpy()
		sel_agg_logit = sel_agg_logit.data.cpu().numpy()
		where_col_logit = where_col_logit.data.cpu().numpy()
		where_op_logit = where_op_logit.data.cpu().numpy()
		where_start_logit = where_start_logit.data.cpu().numpy()
		where_end_logit = where_end_logit.data.cpu().numpy()

		where_conn_pred = where_conn_logit.argmax(-1)
		sel_num_pred = sel_num_logit.argmax(-1)
		where_num_pred = where_num_logit.argmax(-1)

		sel_col_sorted = (-sel_col_logit).argsort(-1)
		where_col_sorted = (-where_col_logit).argsort(-1)

		# sel_col_sorted = (-sel_agg_logit.max(-1)).argsort(-1)
		# where_col_sorted = (-where_end_logit.max(1)).argsort(-1)

		sel_agg_pred = sel_agg_logit.argmax(-1)
		where_op_pred = where_op_logit.argmax(-1)
		where_start_pred = where_start_logit.argmax(1)
		where_end_pred = where_end_logit.argmax(1)

		ret_queries = []
		B = len(where_conn_logit)
		for b in range(B):
			cur_query = {}
			sel_col_idxs = sel_col_sorted[b][: max(1,sel_num_pred[b])].tolist()
			where_col_idxs = where_col_sorted[b][: max(1, where_num_pred[b])].tolist()
			sel_col_agg = sel_agg_pred[b][sel_col_idxs].tolist()

			cur_query['agg'] = sel_col_agg
			cur_query['cond_conn_op'] = where_conn_pred[b]
			cur_query['sel'] = sel_col_idxs
			cur_query['conds'] = []

			for col_idx in where_col_idxs:
				if col_idx >= len(col[b]):
					break
				cond_op = where_op_pred[b][col_idx]
				cond_start = where_start_pred[b][col_idx]
				cond_end = where_end_pred[b][col_idx]
				cons_toks = q[b][cond_start:cond_end + 1]
				cond_str = merge_tokens(cons_toks, raw_q[b])
				cur_query['conds'].append([col_idx, cond_op, cond_str])

			ret_queries.append(cur_query)

		return ret_queries

"""
得到where之间关系的分类logit
:arg: cls_emb (None, 768)
:return cls_emb[:, :3] 3种
"""
def _get_where_conn_logit(cls_emb):
	return cls_emb[:, :3]

"""
得到select的列数目对应的logit
:arg: cls_emb (None, 768)
:return cls_emb[:, :8]
"""
def _get_sel_num_logit(cls_emb):
	return cls_emb[:, 3:8]

"""
得到where的列数目对应的logit
:arg cls_emb (None, 768)
:return cls_emb[:, 8:13]
"""
def _get_where_num_logit(cls_emb):
	return cls_emb[:, 8:13]

"""
得到是否是select的列对应的logit
:arg col_seq (None, max-col-num, 768)
:return col_seq[:, :, 0]
"""
def _get_sel_col_logit(col_seq):
	return col_seq[:, :, 0]

"""
得到selected列的聚合函数logit  6种
:arg col_seq (None, max-col-num, 768)
:return col_seq[:, :, 1: 7]
"""
def _get_sel_agg_logit(col_seq):
	return col_seq[:, :, 1: 7]

"""
得到是否是where中的列对应的Logit
:arg col_seq (None, max-col-num, 768)
:return col_seq[:, :, 7]
"""
def _get_where_col_logit(col_seq):
	return col_seq[:, :, 7]

"""
得到where列的operation logit  4种
:arg col_seq (None, max-col-num, 768)
:return col_seq[:, :, 1: 7]
"""
def _get_where_op_logit(col_seq):
	return col_seq[:, :, 8: 12]

"""
得到where列的value start
:args q_seq (None, max-qlen, 768)
	  max_col_num 
:return q_seq[:, :, :max_col_num]
"""
def _get_where_start_logit(q_seq, max_col_num):
	return q_seq[:, :, :max_col_num]

"""
得到where列的value end
:args q_seq (None, max-qlen, 768)
	  max_col_num 
:return q_seq[:, :, 100:max_col_num]
"""
def _get_where_end_logit(q_seq, max_col_num):
	return q_seq[:, :, 100:100 + max_col_num]


def _get_logits(cls_emb, q_seq, col_seq, max_col_num):
	return _get_where_conn_logit(cls_emb),\
		   _get_sel_num_logit(cls_emb), \
		   _get_where_num_logit(cls_emb),\
		   _get_sel_col_logit(col_seq), \
		   _get_sel_agg_logit(col_seq), \
		   _get_where_col_logit(col_seq),\
		   _get_where_op_logit(col_seq),\
		   _get_where_start_logit(q_seq, max_col_num),\
		   _get_where_end_logit(q_seq, max_col_num)


def merge_tokens(tok_list, raw_tok_str):
	tok_str = raw_tok_str  # .lower()
	alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
	special = {'-LRB-': '(',
			   '-RRB-': ')',
			   '-LSB-': '[',
			   '-RSB-': ']',
			   '``': '"',
			   '\'\'': '"',
			   '--': u'\u2013'}
	ret = ''
	double_quote_appear = 0
	for raw_tok in tok_list:
		if not raw_tok:
			continue
		tok = special.get(raw_tok, raw_tok)
		if tok == '"':
			double_quote_appear = 1 - double_quote_appear
		if len(ret) == 0:
			pass
		elif len(ret) > 0 and ret + ' ' + tok in tok_str:
			ret = ret + ' '
		elif len(ret) > 0 and ret + tok in tok_str:
			pass
		elif tok == '"':
			if double_quote_appear:
				ret = ret + ' '
		# elif tok[0] not in alphabet:
		#     pass
		elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) \
				and (ret[-1] != '"' or not double_quote_appear):
			ret = ret + ' '
		ret = ret + tok
	return ret.strip()