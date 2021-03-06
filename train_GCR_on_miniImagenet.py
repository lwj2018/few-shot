import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.mini_imagenet import MiniImageNet
from datasets.samplers import CategoriesSampler_train_100way, CategoriesSampler_val_100way
from models.GCR import GCR
from models.convnet import gcrConvnet
from utils.ioUtils import *
from utils.critUtils import loss_for_gcr
from utils.trainUtils import train_gcr
from utils.testUtils import eval_gcr
from torch.utils.tensorboard import SummaryWriter
from utils.dataUtils import getDataloader
from Arguments import Arguments

# Hyper params 
epochs = 500
learning_rate = 1e-5
# Options
shot = 5
dataset = 'miniImage'
store_name = dataset + '_GCR' + '_%dshot'%(shot)
summary_name = 'runs/' + store_name
cnn_ckpt = '/home/liweijie/projects/few-shot/checkpoint/20200329/CNN_best.pth.tar'
reg_ckpt = None
global_ckpt = '/home/liweijie/projects/few-shot/checkpoint/20200329/global_proto_best.pth'
checkpoint = None
log_interval = 20
device_list = '1'
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
train_loader, val_loader = getDataloader(dataset,args)

model_cnn = gcrConvnet().to(device)
# Resume model
if cnn_ckpt is not None:
    resume_cnn_part(model_cnn,cnn_ckpt)
if reg_ckpt is not None:
    resume_model(model_reg,reg_ckpt)
if checkpoint is not None:
    start_epoch, best_acc = resume_gcr_model(model, checkpoint, args.n_base)
global_proto = torch.load(global_ckpt)
global_proto = global_proto[:args.num_class,:]
global_base = torch.Tensor(global_proto[:args.n_base,:]).to(device)
global_novel = torch.Tensor(global_proto[args.n_base:,:]).to(device)
model = GCR(model_cnn,global_base=global_base,global_novel=global_novel,train_way=args.train_way,test_way=args.test_way,\
    shot=args.shot,query=args.query,query_val=args.query_val).to(device)

global_base.requires_grad = True
global_novel.requires_grad = True

# Create loss criterion & optimizer
criterion = loss_for_gcr()

policies = model.get_optim_policies(learning_rate)
optimizer = torch.optim.SGD(policies, momentum=0.9)
optimizer_cnn = torch.optim.SGD(model.baseModel.parameters(), lr=learning_rate,momentum=0.9)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
lr_scheduler_cnn = torch.optim.lr_scheduler.MultiStepLR(optimizer_cnn, milestones=[30,60], gamma=0.1)

# Start training
print("Training Started".center(60, '#'))
for epoch in range(start_epoch, epochs):
    # Train the model
    train_gcr(model,criterion,optimizer,optimizer_cnn,train_loader,device,epoch,log_interval,writer,args)
    # Eval the model
    acc = eval_gcr(model,criterion,val_loader,device,epoch,log_interval,writer,args)
    # Save model
    # remember best acc and save checkpoint
    is_best = acc>best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best': best_acc,
        'global_proto': torch.cat([model.global_base,model.global_novel])
    }, is_best, model_path, store_name)
    print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

print("Training Finished".center(60, '#'))