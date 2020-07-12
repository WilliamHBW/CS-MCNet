import torch as t 
import torch.nn as nn
from collections import OrderedDict
from .basic_module import BasicModule 
import time 
import torchsnooper

class recblock(BasicModule):
    def __init__(self,cr,size):
        super(recblock,self).__init__()

        self.cr = cr
        self.size = size
        self.fc1 = nn.Linear(int(self.cr*self.size*self.size),self.size*self.size)
        cnn_layers = OrderedDict()
        cnn_layers['conv1'] = nn.Conv2d(1,128,1,1,0)
        cnn_layers['relu1'] = nn.ReLU(inplace=True)
        cnn_layers['conv2'] = nn.Conv2d(128,64,1,1,0)
        cnn_layers['relu2'] = nn.ReLU(inplace=True)
        cnn_layers['conv3'] = nn.Conv2d(64,32,3,1,1)
        cnn_layers['relu3'] = nn.ReLU(inplace=True)
        cnn_layers['conv4'] = nn.Conv2d(32,16,3,1,1)
        cnn_layers['relu4'] = nn.ReLU(inplace=True)
        cnn_layers['conv5'] = nn.Conv2d(16,1,3,1,1)
        self.rec_cnn = nn.Sequential(cnn_layers)

    def forward(self,input):
        b_s = input.size(0)
        x_1 = self.fc1(input).view(b_s,self.size,self.size).unsqueeze_(1)
        output = self.rec_cnn(x_1).view(b_s,self.size,self.size)
        return output

class MCblock(BasicModule):
    def __init__(self,m_matrix,cr,blk_size,ref_size):
        super(MCblock,self).__init__()

        self.cr = cr
        self.m = m_matrix
        self.blk_size = blk_size
        self.ref_size = ref_size
        
        self.rec1 = recblock(self.cr,self.blk_size)
        self.rec2 = recblock(self.cr,self.blk_size)
        self.fc = nn.Linear(self.ref_size*self.ref_size,self.blk_size*self.blk_size)

    def forward(self,input,ref,y):
        b_s = input.size(0)
        x_1 = self.rec1(input)
        x_mc = self.fc(ref.view(b_s,self.ref_size*self.ref_size)).view(b_s,self.blk_size,self.blk_size)
        x_2 = ((x_mc+x_1)/2)
        x_3 = x_2.view(b_s,self.blk_size*self.blk_size).unsqueeze_(2)
        weight = self.m.repeat(b_s,1,1)
        y_mc = t.bmm(weight,x_3).squeeze_(2)
        y_r = y_mc-y
        output = self.rec2(y_r)+x_2
        return output,x_mc
    
    def pre_load(self,load_path):
        self.rec1.load(load_path)
        self.rec2.load(load_path)
    
class MCNet(BasicModule):
    def __init__(self,m_matrix,cr,blk_size,ref_size):
        super(MCNet,self).__init__()

        self.m = m_matrix
        self.cr = cr
        self.blk_size = blk_size
        self.ref_size = ref_size

        self.block1 = MCblock(self.m,self.cr,self.blk_size,self.ref_size)
        self.block2 = MCblock(self.m,self.cr,self.blk_size,self.ref_size)
        self.block3 = MCblock(self.m,self.cr,self.blk_size,self.ref_size)
        self.block4 = MCblock(self.m,self.cr,self.blk_size,self.ref_size)

    def forward(self,input,ref,y):
        b_s = input.size(0)
        x_1,x_mc_1 = self.block1(input,ref,y)
        weight = self.m.repeat(b_s,1,1)
        input_2 = t.bmm(weight,x_1.view(b_s,self.blk_size*self.blk_size,1)).squeeze_(2)
        x_2,x_mc_2 = self.block2(input_2,ref,y)
        input_3 = t.bmm(weight,x_2.view(b_s,self.blk_size*self.blk_size,1)).squeeze_(2)
        x_3,x_mc_3 = self.block3(input_3,ref,y)
        input_4 = t.bmm(weight,x_3.view(b_s,self.blk_size*self.blk_size,1)).squeeze_(2)
        output,x_mc_4 = self.block4(input_4,ref,y)
        output_mc = (x_mc_1 + x_mc_2 + x_mc_3 + x_mc_4)/4.0
        return output,output_mc


