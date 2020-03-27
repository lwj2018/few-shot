import os.path as osp

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.mini_imagenet import MiniImageNet
from datasets.samplers import CategoriesSampler_train_100way, CategoriesSampler_val_100way
from models.convnet import Convnet
from models.GCR import Registrator
from utils.ioUtils import *

# Hyper params 
max_epoch = 200
learning_rate = 1e-3
shot = 5
query = 5
query_val = 15
n_base = 80
train_way = 20
test_way = 5
# Options
store_name = 'miniImage_GCR'
cnn_ckpt = None
reg_ckpt = None
global_ckpt = None
log_interval = 100
device_list = '1'
num_workers = 8

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join('runs/miniImage_GCR', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Prepare dataset & dataloader
trainset = MiniImageNet('trainvaltest')
train_sampler = CategoriesSampler_train_100way(trainset.label, 100,
                        train_way, shot, query, n_base)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                        num_workers=num_workers, pin_memory=True)
valset = MiniImageNet('trainvaltest')
val_sampler = CategoriesSampler_val_100way(valset.label, 400,
                        test_way, shot, query_val, n_base)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                        num_workers=num_workers, pin_memory=True)
model_cnn = Convnet().to(device)
model_reg = Registrator().to(device)

# Resume model
if cnn_ckpt is not None:
    resume_model(model_cnn,cnn_ckpt)
if reg_ckpt is not None:
    resume_model(model_reg,reg_ckpt)
global_proto = torch.load(global_ckpt)
global_base = global_proto[:n_base,:]
global_novel = global_proto[n_base:,:]

global_base.requires_grad = True
global_novel.requires_grad = True

optimizer_cnn = torch.optim.SGD(model_cnn.parameters(), lr=learning_rate,momentum=0.9)
optimizer_reg = torch.optim.SGD(model_reg.parameters(), lr=learning_rate,momentum=0.9)
optimizer_global1 = torch.optim.SGD(global_base, lr=learning_rate,momentum=0.9)
optimizer_global2 = torch.optim.SGD(global_novel, lr=learning_rate,momentum=0.9)

lr_scheduler_cnn = torch.optim.lr_scheduler.MultiStepLR(optimizer_cnn, milestones=[30,60], gamma=0.1)
lr_scheduler_reg = torch.optim.lr_scheduler.MultiStepLR(optimizer_atten, milestones=[30,60], gamma=0.1)
lr_scheduler_global1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_global1, milestones=[30,60], gamma=0.1)
lr_scheduler_global2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_global2, milestones=[30,60], gamma=0.1)