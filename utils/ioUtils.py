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

def save_global_proto(state, is_best, model_path):
    torch.save(state, '%s/global_proto_ckpt.pth' % (model_path))
    if is_best:
        shutil.copyfile('%s/global_proto_ckpt.pth' % (model_path),
            '%s/global_proto_best.pth' % (model_path))

def resume_model(model, checkpoint):
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    model.load_state_dict(state_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load model from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
    return params_dict['epoch'], params_dict['best']