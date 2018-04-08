
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np

import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn.init as Init

from PrimaryLayer import PrimaryLayer
from RouteLayer import RouteLayer
from Decoder import Decoder


# In[4]:


class CapsNet(nn.Module):
    def __init__(self,is_cuda):
        super(CapsNet,self).__init__()
        self.is_cuda = is_cuda
        
        self.conv_1 = nn.Conv2d(3,64,5)
        self.conv_2 = nn.Conv2d(64,64,5)
        self.conv_3 = nn.Conv2d(64,128,1)
        self.conv_4 = nn.Conv2d(128,128,5)
        self.conv_5 = nn.Conv2d(128,256,1)
        
        #TParameters need to be Tuned
        self.primary = PrimaryLayer(in_channels=256,out_channles=256,
                                    kernel_size=9,stride=2,caps_dims=8)
        
        self.route_1 = RouteLayer(64,12,1152,8,routing_iter=3,is_cuda=self.is_cuda)
        
        self.route_2 = RouteLayer(10,16,64,12,routing_iter=3,is_cuda=self.is_cuda)
        
        self.Decoder = Decoder(16,1024,self.is_cuda)
        
    def forward(self,x):
        h = F.relu(self.conv_1(x))
        h = F.relu(self.conv_2(h))
        h = F.relu(self.conv_3(h))
        h = F.relu(self.conv_4(h))
        h = F.relu(self.conv_5(h))
        
        # u {b_s,2592,1,8,1}
        u = self.primary(h)
        
        v_1 = self.route_1(u)
        v_2 = self.route_2(v_1)
        #v_2[b_s,10,1,24,1]
        return v_2
        
    def margin_loss(self,v,target,alpha = 0.5,max_l = 0.9,min_l = 0.1):
        batch_size = v.size(0)
        v_norm = torch.sqrt(torch.sum(v**2,dim=-2)).squeeze()
        
        zeros = Variable(torch.zeros(v_norm.size()))
        if self.is_cuda:
            zeros = zeros.cuda()
            target = target.cuda()
        
        upper = torch.max(zeros,max_l-v_norm)**2
        lower = torch.max(zeros,v_norm-min_l)**2
        
        L_c = torch.mul(target,upper) + alpha*torch.mul(1-target,lower)
        margin_loss = torch.mean(torch.sum(L_c,dim=1))
        return margin_loss
    
    def reconstruction_loss(self,reconstruction,image,beta=0.0005):
        batch_size = image.size(0)
        image = torch.mean(image,dim=1).view(batch_size,-1)
        reconstruction_loss = torch.sum((reconstruction - image) ** 2,dim=1)
        reconstruction_loss = beta*torch.mean(reconstruction_loss)
        return reconstruction_loss
    
    def loss(self,v,target,image,alpha=0.5,max_l=0.9,min_l=0.1,
                             use_reconstruction=True,beta=0.0005):
        batch_size = image.size(0)
        margin_loss = self.margin_loss(v,target,alpha,max_l,min_l)
        
        if use_reconstruction:
            reconstruction = self.Decoder(v,target)
            reconstruction_loss = self.reconstruction_loss(reconstruction,image,beta)
            loss = margin_loss+reconstruction_loss
        else:
            loss = margin_loss
            reconstruction_loss = 0
        return loss,margin_loss,reconstruction_loss
        

