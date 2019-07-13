import torch
import torch.nn as nn
import torch.nn.functional as F
logit = torch.tensor([
						[[5,-1], [3,4], [4,5]],
						[[3,1], [1,2], [2,5]],
					], dtype=torch.float)
label = torch.tensor([
	[0, 0, -1],
	[0, 1, 1]
], dtype=torch.long)
ce = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
ce = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
loss1 = F.cross_entropy(logit.transpose(1, 2), label, ignore_index=-1, reduction='mean')
loss2 = F.cross_entropy(logit.transpose(1, 2), label, ignore_index=-1, reduction='none').mean()
print(loss1*5, loss2*6)
