
# coding: utf-8

# In[7]:


import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import numpy as np

import torch.utils.data as Data
import torch.nn.functional as func


# In[8]:


def softmax(input, dim=1):
    """
    nn.functional.softmax does not take a dimension as of PyTorch version 0.2.0.
    This was created to add dimension support to the existing softmax function
    for now until PyTorch 0.4.0 stable is release.
    GitHub issue tracking this: https://github.com/pytorch/pytorch/issues/1020
    Arguments:
        input (Variable): input
        dim (int): A dimension along which softmax will be computed.
    """
    input_size = input.size()

    trans_input = input.transpose(dim, len(input_size) - 1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = func.softmax(input_2d)
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(dim, len(input_size) - 1)


# In[9]:


def one_hot_encode(target, length):
    """Converts batches of class indices to classes of one-hot vectors."""
    batch_s = target.size(0)
    one_hot_vec = torch.zeros(batch_s, length)

    for i in range(batch_s):
        one_hot_vec[i, target[i]] = 1.0

    return one_hot_vec


# In[10]:


def routing(u_hat,b_IJ,r=3):
    b_s = u_hat.size(0)
    l_caps = u_hat.size(1)
    l_plus_caps = u_hat.size(2)
    
    for i in range(r):
        c_IJ = softmax(b_IJ,dim=2) #[b_s,l_caps,l_plus_caps,1,1]
        s_J = torch.sum(torch.mul(u_hat,c_IJ),dim=1,keepdim=True) #[b_s,1,l_plus_caps,l_plus_dims,1]
        v_J = squash(s_J,axis = -2)
        
        v_J_tiled = v_J.repeat(1,u_hat.size(1),1,1,1)
        agreement = torch.matmul(u_hat.view(u_hat.size(0),
                                            u_hat.size(1),
                                            u_hat.size(2),1,-1),v_J_tiled)
        b_IJ = torch.add(b_IJ,agreement)
    return v_J


# In[ ]:


def squash(x, axis = -2):
    square_norm = torch.sum(x**2,dim=axis,keepdim=True)
    safe_norm = torch.sqrt(square_norm)
    unit_vec = x/safe_norm
    squash_factor = square_norm/(1. + square_norm)
    return squash_factor*unit_vec
    

