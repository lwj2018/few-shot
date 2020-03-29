import os.path as osp
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.mini_imagenet import MiniImageNet
from models.convnet import Convnet
from utils.ioUtils import *
from utils.trainUtils import train_cnn
from utils.testUtils import eval_cnn
from torch.utils.tensorboard import SummaryWriter
from datasets.samplers import CategoriesSampler_train_100way, CategoriesSampler_val_100way


class Arguments:
    def __init__(self):
        self.num_class = 600
        self.shot = 5
        self.query = 5
        self.query_val = 10
        self.n_base = 80
        self.train_way = 20
        self.test_way = 5
        self.feature_dim = 1600
# Hyper params 
epochs = 1000
learning_rate = 1e-5
# Options
store_name = 'CNN'
checkpoint = '/home/liweijie/projects/few-shot/checkpoint/20200329/CNN_best.pth.tar'
log_interval = 20
device_list = '0'
num_workers = 8
model_path = "./checkpoint"

best_acc = 0.00
start_epoch = 0

# Get args
args = Arguments()
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join('runs/cnn', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# Prepare dataset & dataloader
trainset = MiniImageNet('trainvaltest')
train_sampler = CategoriesSampler_train_100way(trainset.label, 100,
                        args.train_way, args.shot, args.query, args.n_base)
train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                        num_workers=num_workers, pin_memory=True)
valset = MiniImageNet('trainvaltest')
val_sampler = CategoriesSampler_val_100way(valset.label, 100,
                        args.test_way, args.shot, args.query_val)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                        num_workers=num_workers, pin_memory=True)
model = Convnet(out_dim=args.num_class, feature_dim=args.feature_dim).to(device)
# Resume model
if checkpoint is not None:
    start_epoch, best_acc = resume_model(model, checkpoint)
# Create loss criterion & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Start training
print("Training Started".center(60, '#'))
for epoch in range(start_epoch, epochs):
    # Train the model
    global_proto = train_cnn(model, criterion, optimizer, train_loader, device, epoch, log_interval, writer, args)
    # Eval the model
    acc = eval_cnn(model, criterion, val_loader, device, epoch, log_interval, writer, args)
    # Save model
    # remember best acc and save checkpoint
    is_best = acc>best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best': best_acc
    }, is_best, model_path, store_name)
    save_global_proto(global_proto, is_best, model_path)
    print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

print("Training Finished".center(60, '#'))