import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.mini_imagenet import MiniImageNet
from datasets.samplers import CategoriesSampler_train_100way, CategoriesSampler_val_100way
from models.GCR_ri import GCR_ri
from models.convnet import gcrConvnet
from utils.ioUtils import *
from utils.critUtils import loss_for_gcr, loss_for_gcr_relation
from utils.trainUtils import train_gcr_relation
from utils.testUtils import eval_gcr_relation
from torch.utils.tensorboard import SummaryWriter

class Arguments:
    def __init__(self):
        self.num_class = 100

        # Settings for 5-shot
        self.shot = 5
        self.query = 5
        self.query_val = 15
        # Settings for 1-shot
        # self.shot = 1
        # self.query = 1
        # self.query_val = 5
        
        self.n_base = 80
        self.train_way = 20
        self.test_way = 5
        self.feature_dim = 1600
# Get args
args = Arguments()
# Hyper params 
epochs = 2000
learning_rate = 1e-4
# Options
store_name = 'miniImage_GCR_ri' + '_%dshot'%(args.shot)
cnn_ckpt = '/home/liweijie/projects/few-shot/checkpoint/20200329/CNN_best.pth.tar'
reg_ckpt = None
global_ckpt = '/home/liweijie/projects/few-shot/checkpoint/20200329/global_proto_best.pth'
checkpoint = None
gcrr_ckpt = '/home/liweijie/projects/few-shot/checkpoint/20200403_miniImage_GCR_r_checkpoint.pth.tar'
log_interval = 20
device_list = '1'
num_workers = 8
model_path = "./checkpoint"

start_epoch = 0
best_acc = 0.00
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join('runs/miniImage_gcr_ri', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

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
model_cnn = gcrConvnet().to(device)

model = GCR_ri(model_cnn,train_way=args.train_way,\
    test_way=args.test_way, shot=args.shot,query=args.query,query_val=args.query_val).to(device)
# Resume model
if cnn_ckpt is not None:
    resume_cnn_part(model_cnn,cnn_ckpt)
if reg_ckpt is not None:
    resume_model(model_reg,reg_ckpt)
if checkpoint is not None:
    start_epoch, best_acc = resume_gcr_model(model, checkpoint, args.n_base)
if gcrr_ckpt is not None:
    resume_gcr_part(model, gcrr_ckpt, args.n_base)
global_proto = torch.load(global_ckpt)
global_proto = global_proto[:args.num_class,:]
global_proto = torch.Tensor(global_proto)
global_base = global_proto[:args.n_base,:]
global_base = global_base.detach().cuda()
global_novel = global_proto[args.n_base:,:]
global_novel = global_novel.detach().cuda()
# model = GCR_relation(model_cnn,global_base=global_base,global_novel=global_novel,train_way=args.train_way,\
#     test_way=args.test_way, shot=args.shot,query=args.query,query_val=args.query_val).to(device)

# Create loss criterion & optimizer
criterion = loss_for_gcr_relation()

policies = model.get_finetune_policies(learning_rate)
optimizer = torch.optim.SGD(policies, momentum=0.9)
optimizer_cnn = torch.optim.SGD(model.baseModel.parameters(), lr=learning_rate,momentum=0.9)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
lr_scheduler_cnn = torch.optim.lr_scheduler.MultiStepLR(optimizer_cnn, milestones=[30,60], gamma=0.1)

# Start training
print("Train with global proto integrated, Save global")
print("Training Started".center(60, '#'))
for epoch in range(start_epoch, epochs):
    # Train the model
    train_gcr_relation(model,criterion,optimizer,optimizer_cnn,train_loader,device,epoch,log_interval,writer,args)
    # Eval the model
    acc = eval_gcr_relation(model,criterion,val_loader,device,epoch,log_interval,writer,args)
    # lr_scheduler.step()
    # lr_scheduler_cnn.step()
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