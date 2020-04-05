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
from utils.critUtils import loss_for_gcr_relation
from utils.testUtils import eval_gcr_relation
from torch.utils.tensorboard import SummaryWriter
from utils.dataUtils import getValloader
from Arguments import Arguments

# Options
dataset = 'miniImage'
shot = 5
store_name = 'eval' + dataset + '_GCR_ri' + '_%dshot'%(shot)
summary_name = 'runs/' + store_name
checkpoint = '/home/liweijie/projects/few-shot/checkpoint/miniImage_GCR_ri_5shot_best.pth.tar'#5-shot
# checkpoint = '/home/liweijie/projects/few-shot/checkpoint/20200404_miniImage_GCR_r_1shot_best.pth.tar'#1-shot
log_interval = 20
device_list = '2'
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
val_loader = getValloader(dataset,args)
model_cnn = gcrConvnet().to(device)
model = GCR_ri(model_cnn,train_way=args.train_way,\
    test_way=args.test_way, shot=args.shot,query=args.query,query_val=args.query_val).to(device)

# Resume model
if checkpoint is not None:
    start_epoch, best_acc = resume_gcr_model(model, checkpoint, args.n_base)


# Create loss criterion
criterion = loss_for_gcr_relation()

# Start Evaluation
print("Evaluation Started".center(60, '#'))
for epoch in range(start_epoch, start_epoch+1):
    acc = eval_gcr_relation(model,criterion,val_loader,device,epoch,log_interval,writer,args)
    print('Batch acc on miniImagenet: {:.3f}'.format(acc))

print("Evaluation Finished".center(60, '#'))