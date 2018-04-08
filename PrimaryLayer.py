
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
from utilFuncs import squash


# In[3]:


class PrimaryLayer(nn.Module):
    
    def __init__(self,in_channels=256,out_channles=256,kernel_size=5,stride=1,caps_dims=8):
        super(PrimaryLayer,self).__init__()
        self.in_channels = in_channels
        self.out_channles = out_channles
        self.kernel_size = kernel_size
        self.stride=stride
        self.caps_dims = caps_dims
        
        self.capsules = nn.Conv2d(in_channels=self.in_channels,
                                  out_channels=self.out_channles,
                                  kernel_size=self.kernel_size,
                                  stride=self.stride)
    
    def forward(self,x):
        '''
        input:
            x:[b_s,width,height,channel]
        
        output:
            y:[b_s,capsules_nums,1,capsules_dims,1]
        '''
        batch_size = x.size(0)
        hidden = self.capsules(x)
        reshaped_hidden = hidden.view(batch_size,-1,1,self.caps_dims,1)
        squashed_hidden = squash(reshaped_hidden,axis=-2)
        return squashed_hidden
        

