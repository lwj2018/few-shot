import torch
import torch.nn.functional as F
import time

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

def train(model_cnn, model_reg, global_base, global_novel,
          optimizer_cnn, optimizer_reg, optimizer_global1,
          optimizer_global2, trainloader, device, epoch, 
          log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # Set trainning mode
    model_cnn.train()
    model_reg.train()

    end = time.time()
    for i, batch in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # get the inputs and labels
        data, lab = [.to(device) for _ in batch]

        # forward
        p = args.shot * args.train_way
        data_shot = data[:p]
        data_query = data[p:]
        data_shot = data_shot[:,:3,:]
        data_query = data_query[:,3:,:]
        train_gt = lab[:p].reshape(args.shot, args.train_way)[0,:]

        proto = model_cnn(data_shot)
        proto = proto.reshape(args.shot, args.train_way, -1)


        which_base = torch.gt(train_gt,79)
        # Maybe call it which_base is better
        which_base = args.train_way-torch.numel(train_gt[which_novel])

        if which_base < args.train_way:
            proto_base = proto[:,:which_base,:]
            proto_novel = proto[:,which_base:,:]
            # Synthesis module corresponds to section 3.2 of the thesis
            #TODO For simplicity, don't use hallucinator
            noise = torch.cuda.FloatTensor((args.train_way-which_base)*args.shot, noise_dim).normal_()
            proto_novel_gen = model_gen(proto_novel.reshape(args.shot*(args.train_way-which_base),-1), noise)
            proto_novel_gen = proto_novel_gen.reshape(args.shot, args.train_way-which_base, -1)
            proto_novel_wgen = torch.cat([proto_novel,proto_novel_gen])
            ind_gen = torch.randperm(2*args.shot)
            train_num = np.random.randint(1, args.shot)
            proto_novel_f = proto_novel_wgen[ind_gen[:train_num],:,:]
            weight_arr = np.random.rand(train_num)
            weight_arr = weight_arr/np.sum(weight_arr)
            # Generate a new sample
            proto_novel_f = (torch.from_numpy(weight_arr.reshape(-1,1,1)).type(torch.float).cuda()*proto_novel_f).sum(dim=0)
            proto_base = proto_base.mean(dim=0)
            # Corresponds to episodic repesentations in the thesis
            proto_final = torch.cat([proto_base, proto_novel_f],0)
        else:
            proto_final = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)


        # compute the loss
        loss = criterion(outputs, target)

        # backward & optimize
        loss.backward()
        optimizer.step()

        # compute the metrics
        prec1, prec5 = accuracy(outputs.data, target, topk=(1,5))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update average value
        losses.update(loss.item())
        top1.update(prec1.item())
        top5.update(prec5.item())

        if i % log_interval == log_interval-1:
            info = ('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                    'Prec@5 {top5.val:.3f}% ({top5.avg:.3f}%)'
                    .format(
                        epoch, i, len(trainloader), batch_time=batch_time,
                        data_time=data_time, loss=losses,  top1=top1, top5=top5,
                        lr=optimizer.param_groups[-1]['lr']))
            print(info)
            writer.add_scalar('train loss',
                    losses.avg,
                    epoch * len(trainloader) + i)
            writer.add_scalar('train acc',
                    top1.avg,
                    epoch * len(trainloader) + i)
            # Reset average meters 
            losses.reset()
            top1.reset()
            top5.reset()

