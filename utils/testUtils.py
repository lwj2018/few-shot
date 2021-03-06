import torch
import torch.nn.functional as F
import time
from utils.metricUtils import *
from utils.Averager import AverageMeter
from utils.Recorder import Recorder
from utils.fewshotUtils import create_nshot_task_label
from collections import OrderedDict
        
def eval_cnn(model, criterion, valloader, 
        device, epoch, log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    averagers = [losses, avg_acc]
    names = ['val loss','val acc']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        # measure data loading time
        recoder.data_tok()

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
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item(),acc]
        recoder.update(vals)

        # logging
        if i==0 or i % log_interval == log_interval-1 or i==len(valloader)-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')
        
    return recoder.get_avg('val acc')

def eval_gcr(model, criterion,
          valloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss1 = AverageMeter()
    avg_loss2 = AverageMeter()
    avg_acc1 = AverageMeter()
    avg_acc2 = AverageMeter()
    # Create recorder
    averagers = [avg_loss1, avg_loss2, avg_acc1, avg_acc2]
    names = ['val loss1','val loss2','val acc1','val acc2']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        p = args.shot * args.test_way
        data_shot = data[:p]
        data_query = data[p:]
        data_shot = data_shot[:,3:,:]
        data_query = data_query[:,3:,:]

        logits, label, logits2, gt = \
                model(data_shot,data_query,lab,mode='eval')
        # compute the loss
        loss, loss1, loss2 = criterion(logits, label, logits2, gt)

        # compute the metrics
        acc1 = accuracy(logits, label)[0]
        acc2 = accuracy(logits2, gt)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss1.item(),loss2.item(),acc1,acc2]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')

    return recoder.get_avg('val acc1')

def test_100way(model, criterion,
          valloader, device, epoch, 
          log_interval, writer, args, relation):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    # Create recorder
    averagers = [avg_loss, avg_acc]
    names = ['val loss', 'val acc']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        data_shot = data[:,3:,:]
        proto = model.baseModel(data_shot)
        global_set = torch.cat([model.global_base,model.global_novel])
        logits = relation(proto,global_set)
        
        # compute the loss
        loss = criterion(logits, lab)

        # compute the metrics
        acc = accuracy(logits, lab)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item(),acc]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Test')

    return recoder.get_avg('val acc')

def eval_gcr_relation(model, criterion,
          valloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss1 = AverageMeter()
    avg_loss2 = AverageMeter()
    avg_loss3 = AverageMeter()
    avg_acc1 = AverageMeter()
    avg_acc2 = AverageMeter()
    avg_acc3 = AverageMeter()
    # Create recorder
    averagers = [avg_loss1, avg_loss2, avg_loss3, avg_acc1, avg_acc2, avg_acc3]
    names = ['val loss1','val loss2','val loss3', 'val acc1','val acc2','val acc3']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        p = args.shot * args.test_way
        data_shot = data[:p]
        data_query = data[p:]
        data_shot = data_shot[:,3:,:]
        data_query = data_query[:,3:,:]

        logits, label, logits2, gt, logits3, gt3 = \
                model(data_shot,data_query,lab,mode='eval')
        # compute the loss
        loss, loss1, loss2, loss3 = criterion(logits, label, logits2, gt, logits3, gt3)

        # compute the metrics
        acc1 = accuracy(logits, label)[0]
        acc2 = accuracy(logits2, gt)[0]
        acc3 = accuracy(logits3, gt3)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss1.item(),loss2.item(),loss3.item(),acc1,acc2,acc3]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')

    return recoder.get_avg('val acc1')

def eval_mn_pn(model, criterion,
          valloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    # Create recorder
    averagers = [avg_loss,avg_acc]
    names = ['val loss','val acc']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    for i, batch in enumerate(valloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        p = args.shot * args.test_way
        data_shot = data[:p]
        data_query = data[p:]
        data_shot = data_shot[:,3:,:]
        data_query = data_query[:,3:,:]

        y_pred, label = model(data_shot,data_query,mode='eval')
        # compute the loss
        loss = criterion(y_pred, label)

        # compute the metrics
        acc = accuracy(y_pred, label)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item(),acc]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')

    return recoder.get_avg('val acc')

def eval_maml(model, criterion,
          valloader, device, epoch, 
          log_interval, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    # Create recorder
    averagers = [avg_loss, avg_acc]
    names = ['val loss','val acc']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)
    # Set evaluation mode
    model.eval()

    recoder.tik()
    recoder.data_tik()
    # Settings
    create_graph = (True if args.order == 2 else False)
    for i, batch in enumerate(valloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs and labels
        data, lab = [_.to(device) for _ in batch]

        # forward
        # data = data.view( ((args.shot+args.query),args.train_way) + data.size()[-3:] )
        # data = data.permute(1,0,2,3,4).contiguous()
        # data = data.view( (-1,) + data.size()[-3:] )
        p = args.shot * args.test_way
        data_shot = data[:p]
        data_query = data[p:]
        data_shot = data_shot[:,:3,:]
        data_query = data_query[:,3:,:]
        data_shape = data.size()[-3:]


        # Create a fast model using the current meta model weights
        fast_weights = OrderedDict(model.named_parameters())

        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(args.inner_train_steps):
            # Perform update of model weights
            y = create_nshot_task_label(args.test_way, args.shot).to(device)
            logits = model.functional_forward(data_shot, fast_weights)
            loss = criterion(logits, y)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            # Update weights manually
            fast_weights = OrderedDict(
                (name, param - args.inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
        
        # Do a pass of the model on the validation data from the current task
        y = create_nshot_task_label(args.test_way, args.query_val).to(device)
        logits = model.functional_forward(data_query, fast_weights)
        loss = criterion(logits, y)
        loss.backward(retain_graph=True)

        # Get post-update accuracies
        y_pred = logits.softmax(-1)
        acc = accuracy(y_pred, y)[0]

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item(),acc]
        recoder.update(vals)

        if i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(valloader),mode='Eval')

    return recoder.get_avg('val acc')
