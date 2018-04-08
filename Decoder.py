
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from utilFuncs import one_hot_encode

# In[ ]:


class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,is_cuda,is_train=True):
        super(Decoder,self).__init__()
        self.is_cuda = is_cuda
        self.is_train = is_train
        self.in_channels = in_channels

        self.fc_1 = nn.Linear(in_channels,128)
        self.fc_2 = nn.Linear(128,512)
        self.fc_3 = nn.Linear(512,1024)
        self.fc_4 = nn.Linear(1024,out_channels)
        
    def forward(self,v,target):
        batch_size = target.size(0)
        target = target.type(torch.FloatTensor)
        if self.is_train:
            mask = torch.stack([target for i in range(self.in_channels)], dim=2)
        else:
            norm = torch.sum(x**2,dim = -2,keepdim=True)
            _,predict = torch.max(norm,dim=1)
            predict = one_hot_encode(predict,10)
            mask = torch.stack([predict for i in range(self.in_channels)], dim=2)
            
        if self.is_cuda:
            mask = mask.cuda()
        
        v_masked = mask*v
        v_masked = torch.sum(v_masked,dim=1)
        
        v = F.relu(self.fc_1(v_masked))
        v = F.relu(self.fc_2(v))
        v = F.relu(self.fc_3(v))
        reconstruction = F.sigmoid(self.fc_4(v))
        
        return reconstruction

