import os
import torch
import shutil

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(state, is_best, model_path, name):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (model_path, name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (model_path, name),
            '%s/%s_best.pth.tar' % (model_path, name))

def resume_model(model, checkpoint):
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    model.load_state_dict(state_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print(params_dict['global_proto'])
    print("Load model from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
    return params_dict['epoch'], params_dict['best']

def resume_gcr(model, checkpoint):
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    model.load_state_dict(state_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load model from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
    return params_dict['epoch'], params_dict['best']#, params_dict['global_proto']

def resume_cnn_part(model, checkpoint):
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    state_dict = {k : v for k,v in state_dict.items() if not 'fc' in k}
    model.load_state_dict(state_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load part of CNN from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
