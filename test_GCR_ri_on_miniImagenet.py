import os.path as osp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.mini_imagenet_drop500 import MiniImageNet2
from datasets.samplers import CategoriesSampler_train_100way, CategoriesSampler_val_100way
from models.GCR_ri import GCR_ri
from models.convnet import gcrConvnet
from utils.ioUtils import *
from utils.critUtils import loss_for_gcr
from utils.testUtils import test_100way
from torch.utils.tensorboard import SummaryWriter

class Arguments:
    def __init__(self):
        self.num_class = 100
        self.shot = 5
        self.query = 5
        self.query_val = 15
        self.n_base = 80
        self.train_way = 20
        self.test_way = 5
        self.feature_dim = 1600
# Hyper params 
epochs = 1000
learning_rate = 1e-3
# Options
checkpoint = '/home/liweijie/projects/few-shot/checkpoint/miniImage_GCR_ri_5shot_best.pth.tar'#5-shot
# checkpoint = '/home/liweijie/projects/few-shot/checkpoint/20200404_miniImage_GCR_r_1shot_best.pth.tar'#1-shot
log_interval = 20
device_list = '2'
num_workers = 8
model_path = "./checkpoint"

start_epoch = 0
best_acc = 0.00
# Get args
args = Arguments()
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join('runs/test_miniImage_gcr_ri', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Prepare dataset & dataloader
valset = MiniImageNet2('trainvaltest')
val_loader = DataLoader(dataset=valset, batch_size = 128,
                        num_workers=8, pin_memory=True, shuffle=True)
valset2 = MiniImageNet2('trainval')
val_loader2 = DataLoader(dataset=valset2, batch_size = 128,
                        num_workers=8, pin_memory=True, shuffle=True)
valset3 = MiniImageNet2('test')
val_loader3 = DataLoader(dataset=valset3, batch_size = 128,
                        num_workers=8, pin_memory=True, shuffle=True)

model_cnn = gcrConvnet().to(device)
model = GCR_ri(model_cnn,train_way=args.train_way,\
    test_way=args.test_way, shot=args.shot,query=args.query,query_val=args.query_val).to(device)
# Resume model
if checkpoint is not None:
    start_epoch, best_acc = resume_gcr_model(model, checkpoint, args.n_base)

# Create loss criterion
criterion = nn.CrossEntropyLoss()

# Start Test
print("Test Started".center(60, '#'))
for epoch in range(start_epoch, start_epoch+1):
    acc = test_100way(model,criterion,val_loader,device,epoch,log_interval,writer,args,model.relation1)
    print('Batch accu_a on miniImagenet: {:.3f}'.format(acc))
    acc = test_100way(model,criterion,val_loader2,device,epoch,log_interval,writer,args,model.relation1)
    print('Batch accu_b on miniImagenet: {:.3f}'.format(acc))
    acc = test_100way(model,criterion,val_loader3,device,epoch,log_interval,writer,args,model.relation1,'test')
    print('Batch accu_n on miniImagenet: {:.3f}'.format(acc))

print("Test Finished".center(60, '#'))