import torch.nn as nn
import torch
import torch.nn.functional as F

class GCR(nn.Module):
    def __init__(self,baseModel):
        super(GCR,self).__init__()
        self.baseModel = baseModel
        self.registrator = Registrator()

    def forward(self, input):
        pass

class Registrator(nn.Module):
    def __init__(self):
        super(Registrator, self).__init__()
        self.fc_params_support = nn.Sequential(
        	torch.nn.Linear(1600, 512),
        	torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
        	)
        self.fc_params_query = nn.Sequential(
        	torch.nn.Linear(1600, 512),
        	torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
        	)

    def forward(self, support_set, query_set):
        support_set_2 = self.fc_params_support(support_set)
        query_set_2 = self.fc_params_query(query_set)
        return support_set_2, query_set_2
