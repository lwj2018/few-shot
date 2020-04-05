from torch.utils.data import DataLoader
from datasets.mini_imagenet import MiniImageNet
from datasets.omniglot import Omniglot
from datasets.samplers import CategoriesSampler_train_100way, CategoriesSampler_val_100way,\
    CategoriesSampler_train, CategoriesSampler_val, CategoriesSampler_train_mn, CategoriesSampler_val_mn

def getDataloader(dataset,args):
    if dataset == 'miniImage':
        trainset = MiniImageNet('trainvaltest')
        train_sampler = CategoriesSampler_train_100way(trainset.label, 100,
                                args.train_way, args.shot, args.query, args.n_base)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                num_workers=args.num_workers, pin_memory=True)
        valset = MiniImageNet('trainvaltest')
        val_sampler = CategoriesSampler_val_100way(valset.label, 100,
                                args.test_way, args.shot, args.query_val)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True)
    elif dataset == 'omniglot':
        trainset = MiniImageNet('trainvaltest')
        train_sampler = CategoriesSampler_train(trainset.label, 100,
                                args.train_way, args.shot, args.query, args.n_base)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                num_workers=args.num_workers, pin_memory=True)
        valset = MiniImageNet('test')
        val_sampler = CategoriesSampler_val(valset.label, 100,
                                args.test_way, args.shot, args.query_val)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader

def getValloader(dataset,args):
    if dataset == 'miniImage':
        valset = MiniImageNet('test')
        val_sampler = CategoriesSampler_val_100way(valset.label, 600,
                                args.test_way, args.shot, args.query_val)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True)
    elif dataset == 'omniglot':
        valset = MiniImageNet('test')
        val_sampler = CategoriesSampler_val(valset.label, 600,
                                args.test_way, args.shot, args.query_val)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True)
    return val_loader

def getMNloader(dataset,args):
    if dataset == 'miniImage':
        trainset = MiniImageNet('trainvaltest')
        train_sampler = CategoriesSampler_train_mn(trainset.label, 100,
                                args.train_way, args.shot, args.query, args.n_base)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                num_workers=args.num_workers, pin_memory=True)
        valset = MiniImageNet('test')
        val_sampler = CategoriesSampler_val_mn(valset.label, 100,
                                args.test_way, args.shot, args.query_val)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True)
    if dataset == 'omniglot':
        trainset = Omniglot('trainvaltest')
        train_sampler = CategoriesSampler_train_mn(trainset.label, 100,
                                args.train_way, args.shot, args.query, args.n_base)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                num_workers=args.num_workers, pin_memory=True)
        valset = Omniglot('test')
        val_sampler = CategoriesSampler_val_mn(valset.label, 100,
                                args.test_way, args.shot, args.query_val)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True)