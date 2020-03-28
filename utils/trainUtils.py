import torch
import torch.nn.functional as F
import numpy
import time
from utils.metricUtils import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def train_cnn(model, criterion, optimizer, trainloader, 
        device, epoch, log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    avg_bleu = AverageMeter()
    avg_proto = AverageMeter()
    # Set trainning mode
    model.train()

    end = time.time()
    for i, batch in enumerate(trainloader,1):
        # measure data loading time
        data_time.update(time.time() - end)

        # get the data and labels
        data,lab = [_.to(device) for _ in batch]

        p = args.shot * args.train_way
        data_shot = data[:p]
        data_query = data[p:]
        data_shot = data_shot[:,:3,:]
        data_query = data_query[:,:3,:]
        input = torch.cat([data_shot,data_query],0)

        optimizer.zero_grad()
        # forward
        outputs = model(input)

        # compute the loss
        loss = criterion(outputs,lab)

        # backward & optimize
        loss.backward()

        optimizer.step()

        # Calculate global proto
        proto = model.get_feature(input)
        episodic_proto = numpy.zeros([args.num_class,args.feature_dim])
        for idx,p in enumerate(proto):
            p = p.data.detach().cpu().numpy()
            c = lab[idx]
            episodic_proto[c] += p
        episodic_proto = episodic_proto/(args.shot+args.query)
        # compute the metrics
        acc = accuracy(outputs, lab)[0]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update average value
        N = lab.size(0)
        losses.update(loss.item(),N)
        avg_acc.update(acc,N)
        avg_proto.update(episodic_proto)

        if i==0 or i % log_interval == log_interval-1:
            info = ('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'acc {acc.val:.3f} ({acc.avg:.3f})\t'
                    .format(
                        epoch, i, len(trainloader), batch_time=batch_time,
                        data_time=data_time, loss=losses,  acc=avg_acc))
            print(info)
            writer.add_scalar('train loss',
                    losses.avg,
                    epoch * len(trainloader) + i)
            writer.add_scalar('train acc',
                    avg_acc.avg,
                    epoch * len(trainloader) + i)
            # Reset average meters 
            losses.reset()
            avg_acc.reset()

    global_proto = avg_proto.avg
    return global_proto

def train(model, global_base, global_novel, criterion,
          optimizer_cnn, optimizer_reg, optimizer_global1,
          optimizer_global2, trainloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss1 = AverageMeter()
    avg_loss2 = AverageMeter()
    avg_acc1 = AverageMeter()
    avg_acc2 = AverageMeter()
    # Set trainning mode
    model_cnn.train()
    model_reg.train()

    end = time.time()
    for i, batch in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        p = args.shot * args.train_way
        data_shot = data[:p]
        data_query = data[p:]
        data_shot = data_shot[:,:3,:]
        data_query = data_query[:,3:,:]

        logits, label, logits2, train_gt = 
                model(global_base,global_novel,data_shot,data_query,lab)
        # compute the loss
        loss, loss1, loss2 = criterion(logits, label, logits2, train_gt)

        # backward & optimize
        optimizer_cnn.zero_grad()
        optimizer_reg.zero_grad()
        optimizer_global1.zero_grad()
        optimizer_global2.zero_grad()
        loss.backward()
        if epoch > 45:
            optimizer_cnn.step()
        optimizer_reg.step()
        optimizer_global1.step()
        optimizer_global2.step()

        # compute the metrics
        acc1 = accuracy(logits, label)
        acc2 = accuracy(logits2, train_gt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update average value
        avg_loss1.update(loss1.item())
        avg_loss2.update(loss2.item())
        avg_acc1.update(acc1.item())
        avg_acc2.update(acc2.item())

        if i % log_interval == log_interval-1:
            info = ('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                    'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                    'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                    'Acc1 {acc1.val:.3f}% ({acc1.avg:.3f}%)\t'
                    'Acc2 {acc2.val:.3f}% ({acc2.avg:.3f}%)'
                    .format(
                        epoch, i, len(trainloader), batch_time=batch_time,
                        data_time=data_time, loss1=avg_loss1, loss2=avg_loss2,
                        acc1=avg_acc2, acc2=avg_acc2)
            print(info)
            writer.add_scalar('train loss1',
                    avg_loss2.avg,
                    epoch * len(trainloader) + i)
            writer.add_scalar('train loss2',
                    avg_loss2.avg,
                    epoch * len(trainloader) + i)
            writer.add_scalar('train acc1',
                    avg_acc2.avg,
                    epoch * len(trainloader) + i)
            writer.add_scalar('train acc2',
                    avg_acc2.avg,
                    epoch * len(trainloader) + i)
            # Reset average meters 
            avg_loss1.reset()
            avg_loss2.reset()
            avg_acc2.reset()
            avg_acc2.reset()

