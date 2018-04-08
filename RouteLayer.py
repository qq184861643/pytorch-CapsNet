
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import numpy as np
from utilFuncs import squash
from utilFuncs import softmax
from utilFuncs import routing
from torch.autograd import Variable


# In[ ]:


class RouteLayer(nn.Module):
    def __init__(self,caps_nums,caps_dims,before_caps_nums,before_caps_dims,routing_iter,is_cuda):
        super(RouteLayer,self).__init__()
        self.before_caps_nums = before_caps_nums
        self.before_caps_dims = before_caps_dims
        self.caps_nums = caps_nums
        self.caps_dims = caps_dims
        self.is_cuda = is_cuda
        self.routing_iter = routing_iter
        
        self.W = nn.Parameter(torch.randn(1,before_caps_nums,caps_nums,caps_dims,before_caps_dims))
        
    def forward(self,x):
        '''
        input:
            x{b_s,before_caps_nums,1,before_caps_dims,1}
            
        output:
            y[b_s,caps_nums,1,caps_dims,1]
        '''
        
        batch_size = x.size(0)
        x_stack = torch.cat([x]*self.caps_nums,dim=2)
        W_stack = torch.cat([self.W]*batch_size,dim=0)
        
        u_hat = torch.matmul(W_stack,x_stack)
        b_IJ = Variable(torch.zeros(1,self.before_caps_nums,self.caps_nums,1,1))
        if self.is_cuda:
            b_IJ = b_IJ.cuda()
        b_IJ = b_IJ.repeat(batch_size,1,1,1,1)
        
        hidden = routing(u_hat,b_IJ,r=self.routing_iter)
        hidden = hidden.view(batch_size,-1,1,self.caps_dims,1)
        return hidden
        

