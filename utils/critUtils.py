import torch.nn as nn
import torch.nn.functional as F

class loss_for_gcr(nn.Module):

    def __init__(self):
        super(loss_for_gcr,self).__init__()

    def forward(self, logits, label, logits2, train_gt):
        loss1 = F.cross_entropy(logits, label)
        loss2 = F.cross_entropy(logits2, train_gt)
        loss = loss1+loss2
        return loss, loss1, loss2