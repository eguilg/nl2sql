import json
from sqlnet.lib.dbengine import DBEngine
import numpy as np
from tqdm import tqdm
from sqlnet.model.sqlbert import SQLBert
import torch
from sqlnet.strPreprocess import *
from fuzzywuzzy import process
from fuzzywuzzy.utils import StringProcessor
import copy
from sqlnet.diff2 import extact_sort
from sqlnet.diff2 import digit_distance_search
from functools import lru_cache
import re

@lru_cache(None)
def my_scorer(t, c):
	return (1 - abs(len(t) - len(c)) / max(len(t), len(c))) * process.default_scorer(t, c)


def my_process(s):
	"""Process string by
		-- removing all but letters and numbers
		-- trim whitespace
		-- force to lower case
		if force_ascii == True, force convert to ascii"""
	# Force into lowercase.
	string_out = StringProcessor.to_lower_case(s)
	# Remove leading and trailing whitespaces.
	string_out = StringProcessor.strip(string_out)
	return string_out


def pos_in_tokens(target_str, tokens, type = None, header = None):
	if not tokens:
		return -1, -1
	tlen = len(target_str)
	copy_target_str = target_str
	q = ''.join(tokens).replace('##', '')
	header = ''.join(header).replace('##','').replace('[UNK]','')
	ngrams = []
	for l in range(max(1, tlen - 25), min(tlen + 5, len(tokens))):
		ngrams.append(l)
	candidates = {}
	unit_flag = 0
	tback_flag = 0
	unit_r = 0
	if type =='real':
		units = re.findall(r'[(（-](.*)',str(header))
		if units:
			unit = units[0]
			#unit_keys = re.findall(r'[百千万亿]{1,}',str(header))
			unit_keys = re.findall(r'百万|千万|万|百亿|千亿|万亿|亿', unit)
			unit_other = re.findall(r'元|米|平|套|枚|册|张|辆|个|股|户|m²|亩|人', unit)
			if unit_keys:
				unit_flag = 1        #col中有[万|百万|千万|亿]单位，
				unit_key = unit_keys[0]
				v, unit_r = chinese_to_digits(unit_key)
				#print('--unit--',unit_key, target_str)
				target_str = target_str + unit_key
				target_str = strPreProcess(target_str)
				target_str = unit_convert(target_str)
				#print('--target_str--', target_str, header)
			elif unit_other:
				unit_flag = 2    #col中有[元|米|平] 单位为个数
			else:
				unit_flag = 3    # 无单位，可能为个数，可能与ques中单位相同
	for l in ngrams:
		cur_idx = 0
		while cur_idx <= len(tokens) - l:
			cur_str = []
			st, ed = cur_idx, cur_idx
			i = st
			while i != len(tokens) and len(cur_str) < l:
				cur_tok = tokens[i]
				cur_str.append(cur_tok)
				i += 1
				ed = i
			cur_str = ''.join(cur_str)
			if '##' in cur_str :
				cur_str = cur_str.replace('##', '')
			if '[UNK]' in cur_str :
				cur_str = cur_str.replace('[UNK]', '')
			if '-' in cur_str :
				cur_str = cur_str.replace('-', '')

			if unit_flag == 1:
				if cur_str == target_str: #ques 无单位 默认为个数 target_str为unit_convert()后的
					cur_str = str(int(cur_str)/unit_r)
					unit_flag = 0 #target_str回到初始状态，
					tback_flag = 0
				# elif cur_str == copy_target_str: #ques 无单位 默认与target_str 相同
				# 	tback_flag = 1 #标志位 默认与target_str 单位相同
				else:
					cur_str = unit_convert(cur_str)

			elif unit_flag == 2:
				cur_str = unit_convert(cur_str)
			elif unit_flag == 3:
				if unit_convert(cur_str) == target_str:
					cur_str = unit_convert(cur_str)
			if type == 'text':
				for item in list(thesaurus_dic.keys()):
					if item in cur_str:
						cur_str_the = re.sub(item,thesaurus_dic[item],cur_str)
						candidates[cur_str_the] = (st, ed)
			candidates[cur_str] = (st, ed)
			cur_idx += 1
	# if tback_flag:
	# 	target_str = copy_target_str

	if list(candidates.keys()) is None or len(list(candidates.keys())) == 0:
		print('-----testnone----',target_str, tokens,ngrams)
		return -1, -1

	target_str = str(target_str).replace('-', '')
	resultsf = process.extract(target_str, list(candidates.keys()), limit=10, processor=my_process, scorer=my_scorer)
	results = extact_sort(target_str, list(candidates.keys()), limit=10)
	if not results or not resultsf:
		return -1, -1
	dchosen, dcscore = results[0]
	fchosen, fcscore = resultsf[0]
	if fcscore > dcscore:
		cscore = fcscore
		chosen = fchosen
	else:
		cscore = dcscore
		chosen = dchosen

	if cscore !=100:
		pass
		#q = ''.join(tokens).replace('##','')
		#score = '%d'%(cscore)
		#with open("F:\\天池比赛\\nl2sql_test_20190618\\log3.txt", "a", encoding='utf-8') as fw:
			#fw.write(str(chosen + '-----' + target_str + '---'+score +'--'+ q +'\n'+'\n'))

	if cscore <= 50:
		q = ''.join(tokens).replace('##','')
		score = '%d'%(cscore)
		#with open("F:\\天池比赛\\nl2sql_test_20190618\\log3.txt", "a", encoding='utf-8') as fw:
			#fw.write(str(type + '  '+ header + ' ** '+chosen + '-----' + target_str + '---'+score +'--'+ q +'\n'+'\n'))
		#return -1, -1
	return candidates[chosen]
	#return cscore, chosen


def justify_col_type(table):
	def get_real_col_type(col_idx):
		ret_type = 'text'
		if 'rows' not in table.keys():
			if 'types' in table.keys():
				ret_type = table['types'][col_idx]
		else:
			na_set = {'None', 'none', 'N/A', '', 'nan', '-', '/', 'NAN'}
			col_data = list(filter(lambda x: x not in na_set, [r[col_idx] for r in table['rows']]))
			if col_data:
				isreal = True
				try:
					_ = list(map(float, col_data))
				except:
					isreal = False
				if isreal:
					ret_type = 'real'
				if ('ISBN' in table['header'][col_idx]) or ('号' in table['header'][col_idx]) or ('ID' in table['header'][col_idx]):
					ret_type = 'text'
				if ret_type != table['types'][col_idx]:
					print(table['header'][col_idx], col_data)

		return ret_type

	if 'types' not in table.keys():
		table['types'] = ['text'] * len(table['header'])
	for i in range(len(table['header'])):
		table['types'][i] = get_real_col_type(i)
	return table


def load_data(sql_paths, table_paths, use_small=False):
	if not isinstance(sql_paths, list):
		sql_paths = (sql_paths,)
	if not isinstance(table_paths, list):
		table_paths = (table_paths,)
	sql_data = []
	table_data = {}

	for SQL_PATH in sql_paths:
		with open(SQL_PATH, encoding='utf-8') as inf:
			for idx, line in enumerate(inf):
				sql = json.loads(line.strip())
				if use_small and idx >= 1000:
					break
				sql_data.append(sql)
		print("Loaded %d data from %s" % (len(sql_data), SQL_PATH))

	for TABLE_PATH in table_paths:
		with open(TABLE_PATH, encoding='utf-8') as inf:
			for line in inf:
				tab = json.loads(line.strip())
				tab = justify_col_type(tab)
				table_data[tab[u'id']] = tab
		print("Loaded %d data from %s" % (len(table_data), TABLE_PATH))

	ret_sql_data = []
	for sql in sql_data:
		if sql[u'table_id'] in table_data:
			ret_sql_data.append(sql)

	return ret_sql_data, table_data


def load_dataset(data_dir='../data', toy=False, use_small=False, mode='train'):
	print("Loading dataset")
	import os.path as osp
	data_dirs = {}
	for name in ['train', 'val', 'test']:
		data_dirs[name] = {}
		data_dirs[name]['data'] = osp.join(data_dir, name, name+'.json')
		data_dirs[name]['tables'] = osp.join(data_dir, name, name + '.tables.json')
		data_dirs[name]['db'] = osp.join(data_dir, name, name + '.db')

	dev_sql, dev_table = load_data(data_dirs['val']['data'], data_dirs['val']['tables'], use_small=use_small)
	dev_db = data_dirs['val']['db']
	if mode == 'train':
		train_sql, train_table = load_data(data_dirs['train']['data'], data_dirs['train']['tables'], use_small=use_small)
		train_db = data_dirs['train']['db']
		return train_sql, train_table, train_db, dev_sql, dev_table, dev_db
	elif mode == 'test':
		test_sql, test_table = load_data(data_dirs['test']['data'], data_dirs['test']['tables'], use_small=use_small)
		test_db = data_dirs['test']['db']
		return dev_sql, dev_table, dev_db, test_sql, test_table, test_db


def to_batch_seq(sql_data, table_data, idxes, st, ed, tokenizer=None, ret_vis_data=False):
	q_seq = []   #问题内容
	col_seq = []  #一张表所有表头
	col_num = []   #表头数量
	ans_seq = []    #sql的答案列表
	gt_cond_seq = []  #条件列--列号，类型，值
	vis_seq = []		#（）tuple，问题和对应表所有表头
	sel_num_seq = []	#sel列的数量
	header_type = []	#对应表所有列的数据类型
	for i in range(st, ed):
		sql = sql_data[idxes[i]]
		sel_num = len(sql['sql']['sel'])
		sel_num_seq.append(sel_num)
		conds_num = len(sql['sql']['conds'])

		if tokenizer:
			q = tokenizer.tokenize(strPreProcess(sql['question']))
			col = [tokenizer.tokenize(header) for header in table_data[sql['table_id']]['header']]

		else:
			q = [char for char in sql['question']]
			col = [[char for char in header] for header in table_data[sql['table_id']]['header']]
		q_seq.append(q)
		col_seq.append(col)
		col_num.append(len(table_data[sql['table_id']]['header']))
		ans_seq.append(
			(
				len(sql['sql']['agg']),
				sql['sql']['sel'],
				sql['sql']['agg'],
				conds_num,
				tuple(x[0] for x in sql['sql']['conds']),
				tuple(x[1] for x in sql['sql']['conds']),
				sql['sql']['cond_conn_op'],
			))
		gt_cond_seq.append(sql['sql']['conds'])
		vis_seq.append((sql['question'], table_data[sql['table_id']]['header']))
		header_type.append(table_data[sql['table_id']]['types'])
	# q_seq: char-based sequence of question
	# gt_sel_num: number of selected columns and aggregation functions
	# col_seq: char-based column name
	# col_num: number of headers in one table
	# ans_seq: (sel, number of conds, sel list in conds, op list in conds)
	# gt_cond_seq: ground truth of conds
	if ret_vis_data:
		return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, header_type, vis_seq
	else:
		return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, header_type


def pad_batch_seqs(seqs, pad=None, max_len=None):
	if not max_len:
		max_len = max([len(s) for s in seqs])
	if not pad:
		pad = 0
	for i in range(len(seqs)):
		if len(seqs[i]) > max_len:
			seqs[i] = seqs[i][:max_len]
		else:
			seqs[i].extend([pad] * (max_len - len(seqs[i])))

	return seqs


def gen_batch_bert_seq(tokenizer, q_seq, col_seq, header_type, max_len=230):
	input_seq = []  # 输入编号
	q_mask = []  # NL mask
	sel_col_mask = []  # columns mask
	sel_col_index = []  # columns starting index
	where_col_mask = []
	where_col_index = []
	token_type_ids = []  # sentence A/B
	attention_mask = []  # length mask

	col_end_index = []

	q_lens = []
	sel_col_nums = []
	where_col_nums = []

	batch_size = len(q_seq)
	for i in range(batch_size):
		text_a = ['[CLS]'] + q_seq[i] + ['[SEP]']
		text_b = []
		for col_idx, col in enumerate(col_seq[i]):
			new_col = []
			if header_type[i][col_idx] == 'text':
				type_token1 = '[unused1]'
				type_token2 = '[unused4]'
				type_token3 = '[unused7]'
			elif header_type[i][col_idx] == 'real':
				type_token1 = '[unused2]'
				type_token2 = '[unused5]'
				type_token3 = '[unused8]'
			else:
				type_token1 = '[unused3]'
				type_token2 = '[unused6]'
				type_token3 = '[unused9]'
			new_col.extend(col)
			new_col.append(type_token2)  # type特征 用来分类第一次作为条件
			new_col.append(type_token3)  # type特征 用来分类第二次作为条件
			# TODO: 可以再加入新的标签来支持更多的列
			new_col.append(type_token1)  # type特征 用来分类sel, 同时分隔列名

			if len(text_a) + len(text_b) + len(new_col) >= max_len:
				break
			text_b.extend(new_col)

		text_b.append('[SEP]')

		inp_seq = text_a + text_b
		input_seq.append(inp_seq)
		q_mask.append([1] * (len(text_a) - 2))
		q_lens.append(len(text_a) - 2)
		token_type_ids.append([0] * len(text_a) + [1] * len(text_b))
		attention_mask.append([1] * len(inp_seq))

		sel_col = []
		where_col = []
		col_ends = []
		for i in range(len(text_a) - 1, len(inp_seq)):
			if inp_seq[i] in ['[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]']:
				where_col.append(i)
			if inp_seq[i] in ['[unused1]', '[unused2]', '[unused3]']:
				sel_col.append(i)
				col_ends.append(i)

		sel_col_mask.append([1] * len(sel_col))
		where_col_mask.append([1] * len(where_col))
		sel_col_nums.append(len(sel_col))
		where_col_nums.append(len(where_col))
		sel_col_index.append(sel_col)
		where_col_index.append(where_col)
		col_end_index.append(col_ends)

	#规范输入为同一长度，pad = ’[pad]‘ | 0
	input_seq = pad_batch_seqs(input_seq, '[PAD]')
	input_seq = [tokenizer.convert_tokens_to_ids(sq) for sq in input_seq] #字符token转化为词汇表里的编码id
	q_mask = pad_batch_seqs(q_mask)
	sel_col_mask = pad_batch_seqs(sel_col_mask)
	sel_col_index = pad_batch_seqs(sel_col_index)
	where_col_mask = pad_batch_seqs(where_col_mask)
	where_col_index = pad_batch_seqs(where_col_index)
	token_type_ids = pad_batch_seqs(token_type_ids)
	attention_mask = pad_batch_seqs(attention_mask)
	col_end_index = pad_batch_seqs(col_end_index)
	return (input_seq, q_mask, sel_col_mask, sel_col_index, where_col_mask, where_col_index, col_end_index,
			token_type_ids, attention_mask), q_lens, sel_col_nums, where_col_nums


def to_batch_seq_test(sql_data, table_data, idxes, st, ed, tokenizer=None):
	q_seq = []
	col_seq = []
	col_num = []
	raw_seq = []
	table_ids = []
	header_type = []
	for i in range(st, ed):
		sql = sql_data[idxes[i]]

		if tokenizer:
			q = tokenizer.tokenize(strPreProcess(sql['question']))
			col = [tokenizer.tokenize(header) for header in table_data[sql['table_id']]['header']]
		else:
			q = [char for char in sql['question']]
			col = [[char for char in header] for header in table_data[sql['table_id']]['header']]
		q_seq.append(q)
		col_seq.append(col)
		col_num.append(len(table_data[sql['table_id']]['header']))
		raw_seq.append(sql['question'])
		table_ids.append(sql_data[idxes[i]]['table_id'])
		header_type.append(table_data[sql['table_id']]['types'])
	return q_seq, col_seq, col_num, raw_seq, table_ids, header_type


def generate_gt_where_seq_test(q, gt_cond_seq):
	ret_seq = []
	for cur_q, ans in zip(q, gt_cond_seq):
		temp_q = u"".join(cur_q)
		cur_q = [u'<BEG>'] + cur_q + [u'<END>']
		record = []
		record_cond = []
		for cond in ans:
			if cond[2] not in temp_q:
				record.append((False, cond[2]))
			else:
				record.append((True, cond[2]))
		for idx, item in enumerate(record):
			temp_ret_seq = []
			if item[0]:
				temp_ret_seq.append(0)
				temp_ret_seq.extend(list(range(temp_q.index(item[1]) + 1, temp_q.index(item[1]) + len(item[1]) + 1)))
				temp_ret_seq.append(len(cur_q) - 1)
			else:
				temp_ret_seq.append([0, len(cur_q) - 1])
			record_cond.append(temp_ret_seq)
		ret_seq.append(record_cond)
	return ret_seq


def gen_bert_labels(q_seq, q_lens, sel_col_nums, where_col_nums, ans_seq, gt_cond_seq, header_type, col_seq):
	q_max_len = max(q_lens)
	sel_col_max_len = max(sel_col_nums)
	where_col_max_len = max(where_col_nums) #2col

	# labels init
	where_conn_label = np.array([x[6] for x in ans_seq])  # (None, )
	sel_num_label = np.array([0 for _ in ans_seq])  # (None, )
	where_num_label = np.array([0 for _ in ans_seq])  # (None, )
	sel_col_label = np.array([[0] * sel_col_max_len for _ in ans_seq], dtype=np.float)  # (None, col_max_len)
	sel_agg_label = np.array([[-1] * sel_col_max_len for _ in ans_seq])  # (None, col_max_len)
	where_col_label = np.array([[0] * where_col_max_len for _ in ans_seq], dtype=np.float)  # (None, 2col_max_len)
	where_op_label = np.array([[-1] * where_col_max_len for _ in ans_seq])  # (None, 2col_max_len)

	where_start_label = np.array([[-1] * where_col_max_len for _ in ans_seq])
	where_end_label = np.array([[-1] * where_col_max_len for _ in ans_seq])
	for b in range(len(gt_cond_seq)): # batch_size
		num_conds = len(gt_cond_seq[b]) # 条件数量
		if num_conds == 0:
			where_col_label[b] = 1.0 / sel_col_nums[b]  # 分散
			mass = 0
		else:
			mass = 1 / num_conds
		col_cond_count = {}
		for cond in gt_cond_seq[b]:
			if cond[0] >= sel_col_nums[b]:
				continue

			if cond[0] in col_cond_count:
				col_cond_count[cond[0]] += 1
			else:
				col_cond_count[cond[0]] = 0

			col_idx = 2 * cond[0] + col_cond_count[cond[0]] % 2
			where_op_label[b][col_idx] = cond[1]
			where_col_label[b][col_idx] += mass
			s, e = pos_in_tokens(cond[2], q_seq[b], header_type[b][cond[0]], col_seq[b][cond[0]])
			if s >= 0:
				s = min(s, q_lens[b] - 1)
				e = min(e - 1, q_lens[b] - 1)
				where_start_label[b][col_idx] = s
				where_end_label[b][col_idx] = e

		if num_conds > 0:
			where_num_label[b] = (where_col_label[b] > 0).sum()

		for b in range(len(ans_seq)):
			_sel = ans_seq[b][1]
			_agg = ans_seq[b][2]
			sel, agg = [], []
			for i in range(len(_sel)):
				if _sel[i] < sel_col_nums[b]:
					sel.append(_sel[i])
					agg.append(_agg[i])
			sel_num_label[b] = len(sel)
			mass = 1 / sel_num_label[b]
			if sel_num_label[b] == 0:
				mass = 1 / sel_col_nums[b]
			sel_col_label[b][sel] = mass
			sel_agg_label[b][sel] = agg

	return where_conn_label, sel_num_label, where_num_label, sel_col_label, sel_agg_label, \
		   where_col_label, where_op_label, where_start_label, where_end_label


def to_batch_query(sql_data, idxes, st, ed):
	query_gt = []
	table_ids = []
	for i in range(st, ed):
		sql_data[idxes[i]]['sql']['conds'] = sql_data[idxes[i]]['sql']['conds']
		query_gt.append(sql_data[idxes[i]]['sql'])
		table_ids.append(sql_data[idxes[i]]['table_id'])
	return query_gt, table_ids


def epoch_train(model, optimizer, batch_size, sql_data, table_data, tokenizer=None):
	model.train()
	perm = np.random.permutation(len(sql_data))
	cum_loss = 0.0
	for st in tqdm(range(len(sql_data) // batch_size + 1)):
		if st * batch_size == len(perm):
			break
		ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
		st = st * batch_size
		if isinstance(model, SQLBert):
			# bert training
			q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, header_type = to_batch_seq(sql_data, table_data,
																								  perm, st, ed,
																								  tokenizer=tokenizer)

			bert_inputs, q_lens, sel_col_nums, where_col_nums = gen_batch_bert_seq(tokenizer, q_seq, col_seq,
																				   header_type)
			logits = model.forward(bert_inputs)  # condconn_logits, condop_logits, sel_agg_logits, q2col_logits

			# gen label
			labels = gen_bert_labels(q_seq, q_lens, sel_col_nums, where_col_nums, ans_seq, gt_cond_seq, header_type, col_seq)
			# q_seq  (12,q_lens) 问题内容
			# q_lens  (12,1)问题长度
			# sel_col_nums (12,1) col 长度
			# where_col_nums (12,1)2col长度
			# ans_seq   [(1, [6], [0], 1, (1,), (2,), 0),] len(agg),sel_col,agg,len(con),con_col,con_type,con_op
			# gt_cond_seq (12,3)条件列--列号，类型，值

			# compute loss
			loss = model.loss(logits, labels, q_lens, sel_col_nums)
		else:

			q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, header_type = to_batch_seq(sql_data, table_data,
																								  perm, st, ed)
			# q_seq: char-based sequence of question
			# gt_sel_num: number of selected columns and aggregation functions
			# col_seq: char-based column name
			# col_num: number of headers in one table
			# ans_seq: (sel, number of conds, sel list in conds, op list in conds)
			# gt_cond_seq: ground truth of conds
			gt_where_seq = generate_gt_where_seq_test(q_seq, gt_cond_seq)
			gt_sel_seq = [x[1] for x in ans_seq]
			score = model.forward(q_seq, col_seq, col_num, gt_where=gt_where_seq, gt_cond=gt_cond_seq,
								  gt_sel=gt_sel_seq,
								  gt_sel_num=gt_sel_num)
			# sel_num_score, sel_col_score, sel_agg_score, cond_score, cond_rela_score

			# compute loss
			loss = model.loss(score, ans_seq, gt_where_seq)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		cum_loss += loss.data.cpu().numpy() * (ed - st)
	return cum_loss / len(sql_data)


def predict_test(model, batch_size, sql_data, table_data, output_path, tokenizer=None):
	model.eval()
	perm = list(range(len(sql_data)))
	fw = open(output_path, 'w', encoding='utf-8')
	for st in tqdm(range(len(sql_data) // batch_size + 1)):
		if st * batch_size == len(perm):
			break
		ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
		st = st * batch_size
		with torch.no_grad():
			if isinstance(model, SQLBert):
				q_seq, col_seq, col_num, raw_q_seq, table_ids, header_type = to_batch_seq_test(sql_data, table_data,
																							   perm, st, ed,
																							   tokenizer=tokenizer)

				bert_inputs, q_lens, sel_col_nums, where_col_nums = gen_batch_bert_seq(tokenizer, q_seq, col_seq,
																					   header_type)
				score = model.forward(bert_inputs, return_logits=False)
				sql_preds = model.gen_query(score, q_seq, col_seq, sql_data, table_data, perm, st, ed)
			else:
				q_seq, col_seq, col_num, raw_q_seq, table_ids, header_type = to_batch_seq_test(sql_data, table_data,
																							   perm, st, ed)
				score = model.forward(q_seq, col_seq, col_num)
				sql_preds = model.gen_query(score, q_seq, col_seq, raw_q_seq)
			sql_preds = post_process(sql_preds, sql_data, table_data, perm, st, ed)
		for sql_pred in sql_preds:
			sql_pred = eval(str(sql_pred))
			fw.writelines(json.dumps(sql_pred, ensure_ascii=False) + '\n')
		# fw.writelines(json.dumps(sql_pred,ensure_ascii=False).encode('utf-8')+'\n')
	fw.close()

def epoch_acc(model, batch_size, sql_data, table_data, db_path, tokenizer=None):
	engine = DBEngine(db_path)
	model.eval()
	perm = list(range(len(sql_data)))
	badcase = 0
	one_acc_num, tot_acc_num, ex_acc_num = 0.0, 0.0, 0.0
	total_error_cases = []
	total_gt_cases = []
	for st in tqdm(range(len(sql_data) // batch_size + 1)):
		ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
		st = st * batch_size

		q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, header_type, raw_data = \
			to_batch_seq(sql_data, table_data, perm, st, ed, tokenizer=tokenizer, ret_vis_data=True)
		query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
		# query_gt: ground truth of sql, data['sql'], containing sel, agg, conds:{sel, op, value}
		raw_q_seq = [x[0] for x in raw_data]  # original question

		# try:
		with torch.no_grad():
			if isinstance(model, SQLBert):
				bert_inputs, q_lens, sel_col_nums, where_col_nums = gen_batch_bert_seq(tokenizer, q_seq, col_seq,
																					   header_type)
				score = model.forward(bert_inputs, return_logits=False)
				pred_queries = model.gen_query(score, q_seq, col_seq, sql_data, table_data, perm, st, ed)
			else:
				score = model.forward(q_seq, col_seq, col_num)
				# generate predicted format
				pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq)

		pred_queries_post = copy.deepcopy(pred_queries)
		pred_queries_post = post_process(pred_queries_post, sql_data, table_data, perm, st, ed)
		one_err, tot_err, error_idxs = check_acc(raw_data, pred_queries_post, query_gt)
		error_cases, gt_cases = gen_batch_error_cases(error_idxs, q_seq, query_gt, pred_queries_post, pred_queries, raw_data)
		total_error_cases.extend(error_cases)
		total_gt_cases.extend(gt_cases)

		# except:
		# 	badcase += 1
		# 	print('badcase', badcase)
		# 	continue
		one_acc_num += (ed - st - one_err)
		tot_acc_num += (ed - st - tot_err)

		# Execution Accuracy
		for sql_gt, sql_pred, tid in zip(query_gt, pred_queries_post, table_ids):
			ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
			try:
				ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'],
										  sql_pred['cond_conn_op'])
			except:
				ret_pred = None
			ex_acc_num += (ret_gt == ret_pred)
	save_error_case(total_error_cases, total_gt_cases)
	return one_acc_num / len(sql_data), tot_acc_num / len(sql_data), ex_acc_num / len(sql_data)


def post_process(pred, sql_data, table_data, perm, st, ed):
	for i in range(st, ed):
		sql = sql_data[perm[i]]
		table = table_data[sql['table_id']]
		for c in range(len(pred[i - st]['conds'])):
			col_idx = pred[i - st]['conds'][c][0]
			col_val = pred[i - st]['conds'][c][2]
			if col_idx > len(table['header']) or col_val == "" or table['types'][col_idx] == 'real':
				continue

			col_data = []
			for r in table['rows']:
				if col_idx < len(r) and r[col_idx] not in {'None', 'none'}:#, 'N/A', '-', '/', ''}:
					col_data.append(r[col_idx])
			if not col_data:
				continue

			is_real = True
			try:
				_ = list(map(float, col_data))
			except:
				is_real = False
			if is_real:
				continue

			score_c = 0
			for item in list(thesaurus_dic.keys()):
				if item in col_val:
					col_val_the = re.sub(item, thesaurus_dic[item], col_val)
					match_c, score_c = process.extractOne(col_val_the, col_data, processor=my_process)

			match, score = process.extractOne(col_val, col_data, processor=my_process)
			if score_c > score:
				match = match_c
				score = score_c

			if score < 30:
				continue
			pred[i - st]['conds'][c][2] = match
	return pred


def check_acc(vis_info, pred_queries, gt_queries):

	tot_err = sel_num_err = agg_err = sel_err = 0.0
	cond_num_err = cond_col_err = cond_op_err = cond_val_err = cond_rela_err = 0.0
	bad_sample_idxs = []
	for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
		good = True
		sel_pred, agg_pred, where_rela_pred = pred_qry['sel'], pred_qry['agg'], pred_qry['cond_conn_op']
		sel_gt, agg_gt, where_rela_gt = gt_qry['sel'], gt_qry['agg'], gt_qry['cond_conn_op']

		if where_rela_gt != where_rela_pred:
			good = False
			cond_rela_err += 1

		if len(sel_pred) != len(sel_gt):
			good = False
			sel_num_err += 1

		pred_sel_dict = {k: v for k, v in zip(list(sel_pred), list(agg_pred))}
		gt_sel_dict = {k: v for k, v in zip(sel_gt, agg_gt)}
		if set(sel_pred) != set(sel_gt):
			good = False
			sel_err += 1
		agg_pred = [pred_sel_dict[x] for x in sorted(pred_sel_dict.keys())]
		agg_gt = [gt_sel_dict[x] for x in sorted(gt_sel_dict.keys())]
		if agg_pred != agg_gt:
			good = False
			agg_err += 1

		cond_pred = pred_qry['conds']
		cond_gt = gt_qry['conds']
		if len(cond_pred) != len(cond_gt):
			good = False
			cond_num_err += 1
		else:
			cond_op_pred, cond_op_gt = {}, {}
			cond_val_pred, cond_val_gt = {}, {}
			for p, g in zip(cond_pred, cond_gt):
				cond_op_pred[p[0]] = p[1]
				cond_val_pred[p[0]] = p[2]
				cond_op_gt[g[0]] = g[1]
				cond_val_gt[g[0]] = g[2]

			if set(cond_op_pred.keys()) != set(cond_op_gt.keys()):
				cond_col_err += 1
				good = False

			where_op_pred = [cond_op_pred[x] for x in sorted(cond_op_pred.keys())]
			where_op_gt = [cond_op_gt[x] for x in sorted(cond_op_gt.keys())]
			if where_op_pred != where_op_gt:
				cond_op_err += 1
				good = False

			where_val_pred = [cond_val_pred[x] for x in sorted(cond_val_pred.keys())]
			where_val_gt = [cond_val_gt[x] for x in sorted(cond_val_gt.keys())]
			if where_val_pred != where_val_gt:
				cond_val_err += 1
				good = False

		if not good:
			tot_err += 1
			bad_sample_idxs.append(b)
	return np.array(
		(sel_num_err, sel_err, agg_err, cond_num_err, cond_col_err, cond_op_err, cond_val_err, cond_rela_err)), tot_err, bad_sample_idxs


def gen_batch_error_cases(error_idxs, q_seq, query_gt, pred_queries_post, pred_queries, raw_data):
	error_cases = []
	gt_cases = []
	for idx in error_idxs:
		single_error = {}
		single_gt = {}
		single_error['q_raw'] = raw_data[idx][0]
		single_error['q_normed'] = strPreProcess(single_error['q_raw'])
		single_gt['q_raw'] = raw_data[idx][0]
		single_gt['q_normed'] = strPreProcess(single_error['q_raw'])
		single_error['cols'] = raw_data[idx][1]
		single_gt['cols'] = raw_data[idx][1]
		single_gt['sql'] = query_gt[idx]
		single_error['sql'] = copy.deepcopy(pred_queries_post[idx])
		for i in range(len(single_error['sql']['conds'])):
			single_error['sql']['conds'][i].append(pred_queries[idx]['conds'][i][2])

		error_cases.append(single_error)
		gt_cases.append(single_gt)
	return error_cases, gt_cases


def save_error_case(error_case, gt_cases, dir='./log/'):
	import os.path as osp
	error_fn = osp.join(dir, 'error_cases.json')
	gt_fn = osp.join(dir, 'gt_cases.json')
	with open(error_fn, "w", encoding='utf-8') as f:
		json.dump(error_case, f, ensure_ascii=False, indent=4)
	with open(gt_fn, "w", encoding='utf-8') as f:
		json.dump(gt_cases, f, ensure_ascii=False, indent=4)


def load_word_emb(file_name):
	print('Loading word embedding from %s' % file_name)
	f = open(file_name)
	ret = json.load(f)
	f.close()
	# ret = {}
	# with open(file_name, encoding='latin') as inf:
	#     ret = json.load(inf)
	#     for idx, line in enumerate(inf):
	#         info = line.strip().split(' ')
	#         if info[0].lower() not in ret:
	#             ret[info[0]] = np.array([float(x) for x in info[1:]])
	return ret


thesaurus_dic = {
	'没有要求': '不限',
	'达标': '合格',
	'不': '否',
	'鄂': '湖北',
	'豫': '河南',
	'皖': '安徽',
	'冀': '河北',
	'inter': '因特尔',
	'samsung': '三星',
	'芒果TV': '湖南卫视',
	'湖南台': '芒果TV',
	'企鹅公司': '腾讯',
	'鹅厂': '腾讯',
	'宁': '南京',
	'Youku': '优酷',
	'荔枝台': '江苏卫视',
	'周一': '星期一',
	'周二': '星期二',
	'周三': '星期三',
	'周四': '星期四',
	'周五': '星期五',
	'周六': '星期六',
	'周日': '星期天',
	'周天': '星期天',
	'电视剧频道': '中央台八套',
	'沪': '上海',
	'闽': '福建',
	'增持': '买入'

}