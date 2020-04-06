import torch.nn as nn
import torch
import torch.nn.functional as F
from models.convnet import Convnet
from models.Hallucinator import Hallucinator

class CNN_GEN(nn.Module):
    
    def __init__(self,out_dim, f_dim=1600):
        super(CNN_GEN,self).__init__()
        self.cnn = Convnet(out_dim)
        self.gen = Hallucinator(f_dim)
        
    def forward(self, x):
        x = self.cnn.get_feature(x)
        z = torch.cuda.FloatTensor(x.size()).normal_()
        x = self.gen(x,z)
        x = self.cnn.fc(x)
        return x

    def get_feature(self, x):
        x = self.cnn.get_feature(x)
        return x