import torch
import torch.nn.functional as F
import time
from utils.metricUtils import *
from utils.Averager import AverageMeter
from utils.Recorder import Recorder
        
def eval_cnn(model, criterion, valloader, 
        device, epoch, log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    avg_bleu = AverageMeter()
    # Set evaluation mode
    model.eval()

    end = time.time()
    for i, batch in enumerate(valloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # get the data and labels
        data,lab = [_.to(device) for _ in batch]

        p = args.shot * args.test_way
        data_shot = data[:p]
        data_query = data[p:]
        data_shot = data_shot[:,3:,:]
        data_query = data_query[:,3:,:]
        input = torch.cat([data_shot,data_query],0)

        # forward
        outputs = model(input)

        # compute the loss
        loss = criterion(outputs,lab)

        # compute the metrics
        acc = accuracy(outputs, lab)[0]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update average value
        N = lab.size(0)
        losses.update(loss.item(),N)
        avg_acc.update(acc,N)

        if i==0 or i % log_interval == log_interval-1:
            info = ('[Eval] Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                    'Acc {acc.avg:.3f}%'
                    .format(
                        epoch, i, len(valloader), batch_time=batch_time,
                        data_time=data_time, acc=avg_acc))
            print(info)
    
    accumulate_info = ('[Test] Epoch: [{0}] [len: {1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Batch Prec {acc.avg:.4f}\t'
                .format(
                    epoch, len(valloader), loss=losses,acc=avg_acc
                    ))
    print(accumulate_info) 
    writer.add_scalar('val acc',
            avg_acc.avg,
            epoch)
    return avg_acc.avg
