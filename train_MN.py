import os.path as osp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.mini_imagenet import MiniImageNet
from datasets.samplers import CategoriesSampler_train_mn, CategoriesSampler_val_mn
from models.convnet import gcrConvnet
from models.MN import MN
from utils.ioUtils import *
from utils.trainUtils import train_mn_pn
from utils.testUtils import eval_mn_pn
from torch.utils.tensorboard import SummaryWriter
from utils.dataUtils import getMNloader
from Arguments import Arguments

# Hyper params 
epochs = 1000
learning_rate = 1e-4
# Options
shot = 5
dataset = 'miniImage'
store_name = dataset + '_MN' + '_%dshot'%(shot)
summary_name = 'runs/' + store_name
cnn_ckpt = '/home/liweijie/projects/few-shot/checkpoint/20200329/CNN_best.pth.tar'
checkpoint = None
log_interval = 20
device_list = '0'
num_workers = 8
model_path = "./checkpoint"

start_epoch = 0
best_acc = 0.00
# Get args
args = Arguments(shot,dataset)
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join(summary_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Prepare dataset & dataloader
train_loader,val_loader = getMNloader(dataset,args)
model_cnn = gcrConvnet().to(device)
model = MN(model_cnn,lstm_input_size=args.feature_dim,train_way=args.train_way,test_way=args.test_way,\
    shot=args.shot,query=args.query,query_val=args.query_val).to(device)

# Resume model
if cnn_ckpt is not None:
    resume_cnn_part(model_cnn,cnn_ckpt)
if checkpoint is not None:
    start_epoch, best_acc = resume_model(model, checkpoint)

# Create loss criterion & optimizer
criterion = nn.CrossEntropyLoss()

policies = model.get_optim_policies(learning_rate)
optimizer = torch.optim.SGD(policies)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)

# Start training
print("Training Started".center(60, '#'))
for epoch in range(start_epoch, epochs):
    # Train the model
    train_mn_pn(model,criterion,optimizer,train_loader,device,epoch,log_interval,writer,args)
    # Eval the model
    acc = eval_mn_pn(model,criterion,val_loader,device,epoch,log_interval,writer,args)
    # Save model
    # remember best acc and save checkpoint
    is_best = acc>best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best': best_acc
    }, is_best, model_path, store_name)
    print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

print("Training Finished".center(60, '#'))