#code for conditional attention(bahadanau attn).

import torch
from torch import nn
import torch.nn.functional as F

self._weight = nn.Parameter(torch.FloatTensor(100,1)) #VARIABLE
self._act = nn.tanh() #VARIABLE
self.lin_1 = nn.Linear(var_1_feat_size, 100, bias=False) #VARIABLE
self.lin_2 = nn.Linear(var_2_feat_size, 100, bias=False) #VARIABLE

def context(var_2, var_1):
'''
var_1 - on which attention is conditioned.
var_2 - on which conditional attention is applied.
'''
  attn_wt = F.softmax((self._act(self.lin_2(var_2) + self.lin_1(var_1))).matmul(self.weight), dim = 1)
  return torch.mul(attn_wt, var_2) #VARIABLE
