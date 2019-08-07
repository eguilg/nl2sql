import torch
from sqlnet.utils import *
from sqlnet.model.sqlbert import SQLBert, BertAdam, BertTokenizer
from torch.optim import Adam
from sqlnet.lookahead import Lookahead
import time

import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', action='store_true', help='Whether use gpu')
	parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
	parser.add_argument('--lr', type=float, default=6e-6, help='base learning rate')
	parser.add_argument('--epoch', type=int, default=100, help='Epoch number')
	parser.add_argument('--toy', action='store_true', help='If set, use small data for fast debugging')

	parser.add_argument('--data_dir', type=str, default='../data/')
	parser.add_argument('--bert_model_dir', type=str, default='../model/chinese-bert_chinese_wwm_pytorch/')
	parser.add_argument('--model_save_path', type=str, default='../model/best_bert_model')

	parser.add_argument('--restore', action='store_true', help='Whether restore trained model')
	parser.add_argument('--restore_model_path', type=str, default='../model/best_bert_model')

	args = parser.parse_args()

	gpu = args.gpu
	batch_size = args.batch_size
	epoch = args.epoch
	lr = args.lr

	if args.toy:
		use_small = True
	else:
		use_small = False

	data_dir = args.data_dir
	bert_model_dir = args.bert_model_dir
	model_save_path = args.model_save_path
	restore = args.restore
	restore_model_path = args.restore_model_path

	# load dataset
	train_sql, train_table, train_db, dev_sql, dev_table, dev_db = load_dataset(data_dir=data_dir, use_small=use_small)
	tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)
	model = SQLBert.from_pretrained(bert_model_dir)

	if restore:
		print("Loading trained model from %s" % restore_model_path)
		model.load_state_dict(torch.load(restore_model_path))

	optimizer = BertAdam(model.parameters(), lr=lr, schedule='warmup_cosine',
						 warmup=1.0 / epoch, t_total=epoch * (len(train_sql) // batch_size + 1))

	# base_opt = BertAdam(model.parameters(), lr=learning_rate, schedule='warmup_cosine',
	#                      warmup=1.0/args.epoch, t_total=args.epoch * (len(train_sql) // batch_size + 1))
	# optimizer = Lookahead(base_opt, k=5, alpha=0.5) # Initialize Lookahead


	# used to record best score of each sub-task
	best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv, best_wr = 0, 0, 0, 0, 0, 0, 0, 0
	best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx, best_wr_idx = 0, 0, 0, 0, 0, 0, 0, 0
	best_lf, best_lf_idx = -0.1, 0
	best_ex, best_ex_idx = -0.1, 0
	best_mean, best_mean_idx = -0.1, 0

	print("#" * 20 + "  Start to Train  " + "#" * 20)
	for i in range(args.epoch):
		print('Epoch %d' % (i + 1))
		# train on the train dataset
		train_loss = epoch_train(model, optimizer, batch_size, train_sql, train_table, tokenizer=tokenizer)
		# evaluate on the dev dataset
		dev_acc = epoch_acc(model, batch_size, dev_sql, dev_table, dev_db, tokenizer=tokenizer)

		# accuracy of each sub-task
		print(
			'Sel-Num: %.3f, Sel-Col: %.3f, Sel-Agg: %.3f, W-Num: %.3f, W-Col: %.3f, W-Op: %.3f, W-Val: %.3f, W-Rel: %.3f' % (
				dev_acc[0][0], dev_acc[0][1], dev_acc[0][2], dev_acc[0][3], dev_acc[0][4], dev_acc[0][5], dev_acc[0][6],
				dev_acc[0][7]))
		# save the best model
		if dev_acc[1] > best_lf:
			best_lf = dev_acc[1]
			best_lf_idx = i + 1

		if dev_acc[2] > best_ex:
			best_ex = dev_acc[2]
			best_ex_idx = i + 1

		if (dev_acc[1] + dev_acc[2]) / 2 > best_mean:
			best_mean = (dev_acc[1] + dev_acc[2]) / 2
			best_mean_idx = i + 1
			torch.save(model.state_dict(), model_save_path)

		# record the best score of each sub-task
		if True:
			if dev_acc[0][0] > best_sn:
				best_sn = dev_acc[0][0]
				best_sn_idx = i + 1
			if dev_acc[0][1] > best_sc:
				best_sc = dev_acc[0][1]
				best_sc_idx = i + 1
			if dev_acc[0][2] > best_sa:
				best_sa = dev_acc[0][2]
				best_sa_idx = i + 1
			if dev_acc[0][3] > best_wn:
				best_wn = dev_acc[0][3]
				best_wn_idx = i + 1
			if dev_acc[0][4] > best_wc:
				best_wc = dev_acc[0][4]
				best_wc_idx = i + 1
			if dev_acc[0][5] > best_wo:
				best_wo = dev_acc[0][5]
				best_wo_idx = i + 1
			if dev_acc[0][6] > best_wv:
				best_wv = dev_acc[0][6]
				best_wv_idx = i + 1
			if dev_acc[0][7] > best_wr:
				best_wr = dev_acc[0][7]
				best_wr_idx = i + 1
		print('Train loss = %.3f' % train_loss)
		print('Dev Logic Form Accuracy: %.3f, Execution Accuracy: %.3f' % (dev_acc[1], dev_acc[2]))
		print('Best Logic Form: %.3f at epoch %d' % (best_lf, best_lf_idx))
		print('Best Execution: %.3f at epoch %d' % (best_ex, best_ex_idx))
		print('Best Mean: %.3f at epoch %d' % (best_mean, best_mean_idx))
		if (i + 1) % 10 == 0:
			print('Best val acc: %s\nOn epoch individually %s' % (
				(best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv),
				(best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx)))
