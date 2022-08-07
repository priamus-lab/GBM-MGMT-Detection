import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as torch_functional
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import itertools
import seaborn as sns
import logging
import time
import sys
sys.path.append("EfficientNet-PyTorch-3D")
from efficientnet_pytorch_3d import EfficientNet3D
import monai
import os
#import onnx

class RSNAClassifier(nn.Module):
    def __init__(self):
        super(RSNAClassifier, self).__init__()
        #self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        #self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.conv_1 = self.conv_block(c_in=192, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.conv_2 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.conv_3 = self.conv_block(c_in=64, c_out=32, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.part1 = nn.Sequential(
            self.conv_block(c_in=192, c_out=128, dropout=0.1, kernel_size=3, stride=2, padding=1),
            self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=2, padding=1),
            self.conv_block(c_in=64, c_out=32, dropout=0.1, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2) #verificare sè sensato adottarlo
        )
        self.part2 = nn.Sequential(
            self.conv_block(c_in=192, c_out=128, dropout=0.1, kernel_size=3, stride=2, padding=1),
            self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=2, padding=1),
            self.conv_block(c_in=64, c_out=32, dropout=0.1, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #self.lastcnn = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=16, stride=1, padding=1) #56 
        self.fc = nn.Linear(4096,2) #4096, 9216
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x1, x2):
        x1 = self.part1(x1)    
        x2 = self.part2(x2)
        
        #print(x1.shape)
        N,_,_,_ = x1.size()
        #x1 = x1.view(-1, 32*16*16)
        #x2 = x2.view(-1, 32*16*16)
        x1 = x1.view(N,-1)
        x2 = x2.view(N,-1)
        #print(x1.shape)
        z = torch.cat((x1,x2),1)
        #print(z.shape)
        z = self.fc(z)
        #z = nn.Linear(z.shape[1],2)(z)
        #print(z.shape)
        
        return z
    
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )        
        return seq_block
    
class RSNAClassifierV2(nn.Module):
    def __init__(self):
        super(RSNAClassifierVersion2, self).__init__()
        #self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        #self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.conv_1 = self.conv_block(c_in=192, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.conv_2 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.conv_3 = self.conv_block(c_in=64, c_out=32, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.part1 = nn.Sequential(
            self.conv_block(c_in=192, c_out=128, dropout=0.1, kernel_size=5, stride=2, padding=1),
            self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=5, stride=2, padding=1),
            self.conv_block(c_in=64, c_out=32, dropout=0.1, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2) #verificare sè sensato adottarlo
        )
        self.part2 = nn.Sequential(
            self.conv_block(c_in=192, c_out=128, dropout=0.1, kernel_size=5, stride=2, padding=1),
            self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=5, stride=2, padding=1),
            self.conv_block(c_in=64, c_out=32, dropout=0.1, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #self.lastcnn = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=16, stride=1, padding=1) #56 
        self.fc1 = nn.Linear(4096,1024) #4096, 9216
        self.fc2 = nn.Linear(1024,64)
        self.fc3 = nn.Linear(64,2)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x1, x2):
        x1 = self.part1(x1)    
        x2 = self.part2(x2)
        
        #print(x1.shape)
        N,_,_,_ = x1.size()
        #x1 = x1.view(-1, 32*16*16)
        #x2 = x2.view(-1, 32*16*16)
        x1 = x1.view(N,-1)
        x2 = x2.view(N,-1)
        #print(x1.shape)
        z = torch.cat((x1,x2),1)
        #print(z.shape)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        #z = nn.Linear(z.shape[1],2)(z)
        #print(z.shape)
        
        return z
    
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )        
        return seq_block

class RSNAClassifierFake3D(nn.Module):
    def __init__(self, is_dw, output_size=2):
        super(RSNAClassifierFake3D, self).__init__()
        #self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        #self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.conv_1 = self.conv_block(c_in=192, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.conv_2 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.conv_3 = self.conv_block(c_in=64, c_out=32, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.is_dw = is_dw
        self.output_size = output_size
        self.part1 = nn.Sequential(
            self.reduction_block(c_in=192, c_out=128, kernel_size=3, stride=2, padding=1), #64 -> 32 -> 16 -> 8 -> 4 -> 2
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=16, c_out=8, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=8, c_out=4, kernel_size=3, stride=2, padding=1),
            #self.reduction_block(c_in=4, c_out=2, kernel_size=3, stride=2, padding=1)
            #nn.MaxPool2d(kernel_size=2, stride=2) #verificare sè sensato adottarlo
        )
        self.part2 = nn.Sequential(
            self.reduction_block(c_in=192, c_out=128, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=16, c_out=8, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=8, c_out=4, kernel_size=3, stride=2, padding=1),
            #self.reduction_block(c_in=4, c_out=2, kernel_size=3, stride=2, padding=1)
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #self.lastcnn = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=16, stride=1, padding=1) #56 
        #self.fc = nn.Linear(4096,2) #4096, 9216
        self.fc = self.fully_conn(dropout=0.1)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x1, x2):
        x1 = self.part1(x1)    
        x2 = self.part2(x2)
        
        #print(x1.shape)
        N,_,_,_ = x1.size()
        #x1 = x1.view(-1, 32*16*16)
        #x2 = x2.view(-1, 32*16*16)
        #x1 = x1.view(N,-1) #usare flatten
        #x2 = x2.view(N,-1)
        x1 = torch.flatten(x1,1)
        x2 = torch.flatten(x2,1)
        #print(x1.shape)
        z = torch.cat((x1,x2),1)
        #print(z.shape)
        z = self.fc(z)
        #z = nn.Linear(z.shape[1],2)(z)
        #print(z.shape)
        
        return z
            
    def reduction_block(self, c_in, c_out, kernel_size, stride, padding):
        if self.is_dw:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False),
                nn.BatchNorm2d(num_features=c_out),
                nn.ReLU()
            )
        else:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(num_features=c_out),
                nn.ReLU()
            )
        return downsample
    
    def fully_conn(self, dropout):
        fc = nn.Sequential(
            #nn.Dropout2d(p=dropout),
            nn.Linear(72,16), #4096, 9216
            nn.ReLU(),
            nn.Linear(16,self.output_size)
        )
        return fc
    
class RSNAClassifierSingleFake3D(nn.Module):
    def __init__(self, is_dw, output_size=2):
        super(RSNAClassifierSingleFake3D, self).__init__()
        self.is_dw = is_dw
        self.output_size = output_size
        self.part1 = nn.Sequential(
            self.reduction_block(c_in=192, c_out=128, kernel_size=3, stride=2, padding=1), #64 -> 32 -> 16 -> 8 -> 4 -> 2
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=16, c_out=8, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=8, c_out=4, kernel_size=3, stride=2, padding=1),
        )
        self.fc = self.fully_conn(dropout=0.1)
        
    def forward(self, x):
        x = self.part1(x)    
        
        #print(x1.shape)
        N,_,_,_ = x.size()
        z = torch.flatten(x,1)
        #print(z.shape)
        z = self.fc(z)
        
        return z
            
    def reduction_block(self, c_in, c_out, kernel_size, stride, padding):
        if self.is_dw:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False),
                nn.BatchNorm2d(num_features=c_out),
                nn.ReLU()
            )
        else:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(num_features=c_out),
                nn.ReLU()
            )
        return downsample
    
    def fully_conn(self, dropout):
        fc = nn.Sequential(
            #nn.Dropout2d(p=dropout),
            nn.Linear(36,16), #4096, 9216
            nn.ReLU(),
            nn.Linear(16,self.output_size)
        )
        return fc
    
class RSNAClassifier3D(nn.Module):
    def __init__(self, is_dw, output_size=2):
        super(RSNAClassifier3D, self).__init__()
        self.is_dw = is_dw
        self.output_size = output_size
        self.part1 = nn.Sequential(
            self.conv_block(c_in=1, c_out=4, kernel_size=3, stride=2, padding=1), #64
            self.conv_block(c_in=4, c_out=8, kernel_size=3, stride=2, padding=1), #64
            self.conv_block(c_in=8, c_out=16, kernel_size=3, stride=2, padding=1), #32
            self.conv_block(c_in=16, c_out=32, kernel_size=3, stride=2, padding=1), #16
            self.conv_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1), #8
            self.conv_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1), #4
            #self.conv_mp_block(c_in=128, c_out=192, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=128, c_out=256, kernel_size=3, stride=2, padding=1), #2
        )
        
        self.part2 = nn.Sequential(
            self.conv_block(c_in=1, c_out=4, kernel_size=3, stride=2, padding=1), #64
            self.conv_block(c_in=4, c_out=8, kernel_size=3, stride=2, padding=1), #64
            self.conv_block(c_in=8, c_out=16, kernel_size=3, stride=2, padding=1), #32
            self.conv_block(c_in=16, c_out=32, kernel_size=3, stride=2, padding=1), #16
            self.conv_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1), #8
            self.conv_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1), #4
            #self.conv_mp_block(c_in=128, c_out=192, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=128, c_out=256, kernel_size=3, stride=2, padding=1), #2
        )
        self.fc = self.fully_conn(dropout=0.1)
        
    def forward(self, x1, x2):
        x1 = self.part1(x1)    
        x2 = self.part2(x2)
        #print(x1.shape)
        x1 = torch.flatten(x1,1)
        x2 = torch.flatten(x2,1)
        #print(x1.shape)
        z = torch.cat((x1,x2),1)
        #print(z.shape)
        z = self.fc(z)
        #print(z.shape)
        return z
            
    def conv_block(self, c_in, c_out, kernel_size, stride, padding):
        if self.is_dw:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
                nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
                nn.BatchNorm3d(num_features=c_out),
                nn.ReLU()
            )
        else:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm3d(num_features=c_out),
                nn.ReLU()
            )
        return downsample
    
    def conv_mp_block(self, c_in, c_out, kernel_size, stride, padding):
        downsample = nn.Sequential(
            nn.Conv3d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
            #nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU()
        )
        return downsample
    
    def conv_us_block(self, c_in, c_out, kernel_size, stride, padding):
        downsample = nn.Sequential(
            nn.Conv3d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
            #nn.Upsample(scale_factor=kernel_size),
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU()
        )
        return downsample
    
    def fully_conn(self, dropout):
        fc = nn.Sequential(
            #nn.Dropout2d(p=dropout),
            nn.Linear(4096,64), #6144/6 - 32768/32 (32768,1024)
            nn.ReLU(),
            #nn.Linear(1024,256), #1024/32
            #nn.ReLU(),
            #nn.Linear(256,64),
            #nn.ReLU(),
            nn.Linear(64,self.output_size)
        )
        return fc  
    
class RSNAClassifierSingle3D(nn.Module):
    def __init__(self, is_dw, output_size=2):
        super(RSNAClassifierSingle3D, self).__init__()
        self.is_dw = is_dw
        self.output_size = output_size
        self.part1 = nn.Sequential(
            self.conv_block(c_in=1, c_out=4, kernel_size=3, stride=2, padding=1), #192=>96
            self.conv_block(c_in=4, c_out=8, kernel_size=3, stride=2, padding=1), #96=>48
            self.conv_block(c_in=8, c_out=16, kernel_size=3, stride=2, padding=1), #48=>24
            self.conv_block(c_in=16, c_out=32, kernel_size=3, stride=2, padding=1), #24=>12
            self.conv_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1), #12=>6
            self.conv_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1), #6=>3
            #self.conv_mp_block(c_in=128, c_out=192, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=128, c_out=256, kernel_size=3, stride=2, padding=1), #3=>1
        )
        
        self.fc = self.fully_conn(dropout=0.1)
        
    def forward(self, x1):
        x1 = self.part1(x1)    
        #print(x1.shape)
        x1 = torch.flatten(x1,1)
        #print(x1.shape)
        z = self.fc(x1)
        #print(z.shape)
        return z
            
    def conv_block(self, c_in, c_out, kernel_size, stride, padding):
        if self.is_dw:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
                nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
                nn.BatchNorm3d(num_features=c_out),
                nn.ReLU()
            )
        else:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm3d(num_features=c_out),
                nn.ReLU()
            )
        return downsample
    """
    def conv_mp_block(self, c_in, c_out, kernel_size, stride, padding):
        downsample = nn.Sequential(
            nn.Conv3d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
            #nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU()
        )
        return downsample
    
    def conv_us_block(self, c_in, c_out, kernel_size, stride, padding):
        downsample = nn.Sequential(
            nn.Conv3d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
            #nn.Upsample(scale_factor=kernel_size),
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU()
        )
        return downsample
    """
    
    def fully_conn(self, dropout):
        fc = nn.Sequential(
            #nn.Dropout2d(p=dropout),
            nn.Linear(2048,64), #6144/6 - 32768/32 (32768,1024)
            nn.ReLU(),
            #nn.Linear(512,128), #1024/32
            #nn.ReLU(),
            #nn.Linear(128,32),
            #nn.ReLU(),
            nn.Linear(64,self.output_size)
        )
        return fc  
    
class RSNAClassifierSingleVoting2D(nn.Module):
    def __init__(self, is_dw, output_size=2):#, sel_slices=32):
        super(RSNAClassifierSingleVoting2D, self).__init__()
        self.is_dw = is_dw
        self.output_size = output_size
        #self.sel_slices = sel_slices
        #self.convs = []
        #self.fcs = []
        """
        for sl in range(sel_slices):
            conv = nn.Sequential(
                self.reduction_block(c_in=1, c_out=32, kernel_size=3, stride=2, padding=1),
                self.reduction_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1),
                self.reduction_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1),
                self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
                self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
                self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
            )
            self.convs.append(conv)
            fc = self.fully_conn(dropout=0.1)
            self.fcs.append(fc)
        """
        self.conv = nn.Sequential(
            self.reduction_block(c_in=1, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
        )
        self.fc = self.fully_conn(dropout=0.1)
        """
        self.part1 = nn.Sequential(
            self.reduction_block(c_in=1, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
        )
        self.part2 = nn.Sequential(
            self.reduction_block(c_in=1, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
        )
        self.part3 = nn.Sequential(
            self.reduction_block(c_in=1, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
        )
        self.part4 = nn.Sequential(
            self.reduction_block(c_in=1, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
        )
        self.part5 = nn.Sequential(
            self.reduction_block(c_in=1, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
        )
        self.part6 = nn.Sequential(
            self.reduction_block(c_in=1, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
        )
        self.part7 = nn.Sequential(
            self.reduction_block(c_in=1, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
        )
        self.part8 = nn.Sequential(
            self.reduction_block(c_in=1, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=128, c_out=64, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=64, c_out=32, kernel_size=3, stride=2, padding=1),
            self.reduction_block(c_in=32, c_out=16, kernel_size=3, stride=2, padding=1),
        )
        
        self.fc = self.fully_conn(dropout=0.1)
        """
        
    def single_forward(self, x_list, i):
        x_slice = torch.unsqueeze(x_list[:,i,:,:], dim=1)
        x = self.convs[i](x_slice)
        z = torch.flatten(x,1)
        z = self.fcs[i](z)
        return z
        
    def forward(self, x):
        #x_slice = torch.unsqueeze(x_list, dim=1)
        x = self.conv(x)
        z = torch.flatten(x,1)
        z = self.fc(z)
        #z_list = torch.cat([self.single_forward(x_list, i) for i in range(x_list.shape[1])])
        
        """
        x1 = self.part1(x_list[:,0,:,:])
        x2 = self.part1(x_list[:,1,:,:])  
        x3 = self.part1(x_list[:,2,:,:])  
        x4 = self.part1(x_list[:,3,:,:])  
        x5 = self.part1(x_list[:,4,:,:])  
        x6 = self.part1(x_list[:,5,:,:])  
        x7 = self.part1(x_list[:,6,:,:])  
        x8 = self.part1(x_list[:,7,:,:])  
        
        z1 = torch.flatten(x1,1)
        z2 = torch.flatten(x2,1)
        z3 = torch.flatten(x3,1)
        z4 = torch.flatten(x4,1)
        z5 = torch.flatten(x5,1)
        z6 = torch.flatten(x6,1)
        z7 = torch.flatten(x7,1)
        z8 = torch.flatten(x8,1)
        #print(z.shape)
        z1 = self.fc(z1)
        z2 = self.fc(z2)
        z3 = self.fc(z3)
        z4 = self.fc(z4)
        z5 = self.fc(z5)
        z6 = self.fc(z6)
        z7 = self.fc(z7)
        z8 = self.fc(z8)
        """
        #z = torch.mean(z_list)
        
        return z
            
    def reduction_block(self, c_in, c_out, kernel_size, stride, padding):
        if self.is_dw:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False),
                nn.BatchNorm2d(num_features=c_out),
                nn.ReLU()
            )
        else:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(num_features=c_out),
                nn.ReLU()
            )
        return downsample
    
    def fully_conn(self, dropout):
        fc = nn.Sequential(
            #nn.Dropout2d(p=dropout),
            nn.Linear(144,16), #4096, 9216
            nn.ReLU(),
            nn.Linear(16,self.output_size)
        )
        return fc
    
class RSNAAlternativeClassifierSingle2D(nn.Module):
    def __init__(self, is_dw, output_size=2):#, sel_slices=32):
        super(RSNAAlternativeClassifierSingle2D, self).__init__()
        self.is_dw = is_dw
        self.output_size = output_size
        #self.sel_slices = sel_slices
        #self.convs = []
        #self.fcs = []

        self.conv = nn.Sequential(
            self.conv_block(c_in=1, c_out=16, kernel_size=4, stride=1, padding=0),
            self.batch_norm_block(c_out=16),
            self.max_pool_block(kernel_size=4, stride=4),
            self.conv_block(c_in=16, c_out=16, kernel_size=4, stride=1, padding=0),
            self.max_pool_block(kernel_size=2, stride=2),
            self.conv_block(c_in=16, c_out=16, kernel_size=4, stride=1, padding=0),
            self.max_pool_block(kernel_size=2, stride=2),
            self.conv_block(c_in=16, c_out=16, kernel_size=1, stride=1, padding=0),
        )
        self.fc = self.fully_conn(dropout=0.25)
        
    def forward(self, x):
        #x_slice = torch.unsqueeze(x_list, dim=1)
        x = self.conv(x)
        z = torch.flatten(x,1)
        z = self.fc(z)
        #z_list = torch.cat([self.single_forward(x_list, i) for i in range(x_list.shape[1])])
        
        return z
            
    def conv_block(self, c_in, c_out, kernel_size, stride, padding):
        if self.is_dw:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False),
                nn.ReLU()
            )
        else:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.ReLU()
            )
        return downsample
    
    def batch_norm_block(self, c_out):
        return nn.Sequential(
            nn.BatchNorm2d(num_features=c_out)
        )
    
    def max_pool_block(self, kernel_size, stride):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        )
    
    def fully_conn(self, dropout):
        fc = nn.Sequential(
            #nn.Dropout2d(p=dropout),
            nn.Linear(1296,16), #4096, 9216
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16,self.output_size)
        )
        return fc
    
class BinaryEfficientNet3D(nn.Module):
    def __init__(self, output_size=2):
        super().__init__()
        self.is_dw = False
        self.output_size = output_size
        self.net = EfficientNet3D.from_name("efficientnet-b3", override_params={'num_classes': self.output_size}, in_channels=1)
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=self.output_size, bias=True)
    
    def forward(self, x):
        out = self.net(x)
        return out
    
class VoxCNN(nn.Module):
    def __init__(self, output_size=2):
        super(VoxCNN, self).__init__()
        self.is_dw = False
        self.output_size = output_size
        self.conv_1 = nn.Sequential(
            self.conv_block(c_in=1, c_out=8, kernel_size=3, stride=1, padding=0),
            self.conv_block(c_in=8, c_out=8, kernel_size=3, stride=1, padding=0),
            self.pool_block(kernel_size=2, stride=2),
        )
        
        self.conv_2 = nn.Sequential(
            self.conv_block(c_in=8, c_out=16, kernel_size=3, stride=1, padding=0),
            self.conv_block(c_in=16, c_out=16, kernel_size=3, stride=1, padding=0),
            self.pool_block(kernel_size=2, stride=2),
        )
        
        self.conv_3 = nn.Sequential(
            self.conv_block(c_in=16, c_out=32, kernel_size=3, stride=1, padding=0),
            self.conv_block(c_in=32, c_out=32, kernel_size=3, stride=1, padding=0),
            self.conv_block(c_in=32, c_out=32, kernel_size=3, stride=1, padding=0),
            self.pool_block(kernel_size=2, stride=2),
        )
        
        self.conv_4 = nn.Sequential(
            self.conv_block(c_in=32, c_out=64, kernel_size=3, stride=1, padding=0),
            self.conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=0),
            self.conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=0),
            self.pool_block(kernel_size=4, stride=4),
        )
        
        self.fc_all = nn.Sequential(
            self.fc_1(c_in=1728, c_out=128), 
            self.batch_dropout(c_out=128, dropout=0.7),
            self.fc_1(c_in=128, c_out=64),
            self.fc_2(c_in=64, c_out=self.output_size)
        )
        
    def forward(self, x1):
        x1 = self.conv_1(x1) 
        x1 = self.conv_2(x1)  
        x1 = self.conv_3(x1)  
        x1 = self.conv_4(x1)  
        #print(x1.shape)
        x1 = torch.flatten(x1,1)
        #print(x1.shape)
        z = self.fc_all(x1)
        #print(z.shape)
        return z
            
    def conv_block(self, c_in, c_out, kernel_size, stride, padding):
        downsample = nn.Sequential(
            nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU()
        )
        return downsample
    
    def pool_block(self, kernel_size, stride):
        downsample = nn.Sequential(
            nn.MaxPool3d(kernel_size=kernel_size, stride=stride)
        )
        return downsample
    
    def fc_1(self, c_in, c_out):
        downsample = nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.ReLU()
        )
        return downsample
    
    def fc_2(self, c_in, c_out):
        downsample = nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.Softmax(dim=1)
        )
        return downsample
    
    def batch_dropout(self, c_out, dropout):
        downsample = nn.Sequential(
            nn.BatchNorm1d(num_features=c_out),
            nn.Dropout(p=dropout)
        )
        return downsample
    
class RSNAResNet(nn.Module):
    def __init__(self, output_size=2):
        super(RSNAResNet, self).__init__()
        self.is_dw = False
        self.output_size = output_size
        
        self.conv_1 = nn.Sequential(
            self.conv_block(c_in=1, c_out=32, kernel_size=3, stride=1, padding=1),
            self.conv_block(c_in=32, c_out=32, kernel_size=3, stride=1, padding=1),
            self.sim_conv_block(c_in=32, c_out=64, kernel_size=3, stride=2, padding=1),
        )
        
        self.conv_2 = nn.Sequential(
            self.batch_block(c_out=64),
            self.conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
            self.sim_conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
        )
        
        self.conv_3 = nn.Sequential(
            self.batch_block(c_out=64),
            self.conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
            self.sim_conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
        )
        
        self.conv_4 = nn.Sequential(
            self.batch_block(c_out=64),
            self.sim_conv_block(c_in=64, c_out=64, kernel_size=3, stride=2, padding=1),
        )
        
        self.conv_5 = nn.Sequential(
            self.batch_block(c_out=64),
            self.conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
            self.sim_conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
        )
        
        self.conv_6 = nn.Sequential(
            self.batch_block(c_out=64),
            self.conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
            self.sim_conv_block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1),
        )
        
        self.conv_7 = nn.Sequential(
            self.batch_block(c_out=64),
            self.sim_conv_block(c_in=64, c_out=128, kernel_size=3, stride=2, padding=1),
        )
        
        self.conv_8 = nn.Sequential(
            self.batch_block(c_out=128),
            self.conv_block(c_in=128, c_out=128, kernel_size=3, stride=1, padding=1),
            self.sim_conv_block(c_in=128, c_out=128, kernel_size=3, stride=1, padding=1),
        )
        
        self.conv_9 = nn.Sequential(
            self.batch_block(c_out=128),
            self.conv_block(c_in=128, c_out=128, kernel_size=3, stride=1, padding=1),
            self.sim_conv_block(c_in=128, c_out=128, kernel_size=3, stride=1, padding=1),
        )
        
        self.max_pool = nn.Sequential(
            self.pool_block(kernel_size=7, stride=7)
        )
        
        self.fc_all = nn.Sequential(
            self.fc_1(c_in=3456, c_out=128),
            self.fc_2(c_in=128, c_out=self.output_size)
        )
        
    def forward(self, x):
        #x1 = self.conv_1(x) 
        x = self.conv_1(x)
        #x2 = x1 + self.conv_2(x1)
        """
        res = x
        x = self.conv_2(x)
        x += res
        """
        #x3 = x2 + self.conv_3(x2)  
        res = x
        x = self.conv_3(x)
        x += res
        #x4 = self.conv_4(x3)
        x = self.conv_4(x)
        #x5 = x4 + self.conv_5(x4)
        res = x
        x = self.conv_5(x)
        x += res
        #x6 = x5 + self.conv_6(x5)
        res = x
        x = self.conv_6(x)
        x += res
        #x7 = self.conv_7(x6)
        x = self.conv_7(x)
        #x8 = x7 + self.conv_8(x7)
        """
        res = x
        x = self.conv_8(x)
        x += res
        """
        #x9 = x8 + self.conv_9(x8)
        res = x
        x = self.conv_9(x)
        x += res
        #z = self.max_pool(x9)
        z = self.max_pool(x)
        #print(x1.shape)
        z = torch.flatten(z,1)
        #print(x1.shape)
        z = self.fc_all(z)
        #print(z.shape)
        return z
           
    
    def conv_block(self, c_in, c_out, kernel_size, stride, padding):
        downsample = nn.Sequential(
            nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            #nn.BatchNorm3d(num_features=c_out),
            nn.ReLU()
        )
        return downsample
    
    def sim_conv_block(self, c_in, c_out, kernel_size, stride, padding):
        downsample = nn.Sequential(
            nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        )
        return downsample
    
    def batch_block(self, c_out):
        downsample = nn.Sequential(
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU()
        )
        return downsample
    
    def pool_block(self, kernel_size, stride):
        downsample = nn.Sequential(
            nn.MaxPool3d(kernel_size=kernel_size, stride=stride)
        )
        return downsample
        
    def batch_dropout(self, c_out, dropout):
        downsample = nn.Sequential(
            nn.BatchNorm1d(num_features=c_out),
            nn.Dropout(p=dropout)
        )
        return downsample
    
    def fc_1(self, c_in, c_out):
        downsample = nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.ReLU()
        )
        return downsample
    
    def fc_2(self, c_in, c_out):
        downsample = nn.Sequential(
            nn.Linear(c_in, c_out),
            #nn.Softmax(dim=1)
        )
        return downsample
    
    
class TunAIResNet(nn.Module):
    def __init__(self, output_size=2):
        super().__init__()
        self.is_dw = False
        self.output_size = output_size
        self.net = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=self.output_size)
    
    def forward(self, x):
        out = self.net(x)
        return out
    
class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
        return self.double_conv(x)

    
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)

    
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask
"""
class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output
"""

def try_and_get_model(chosen_net, dims, size, is_depth_wise, output_size):
    if chosen_net == "sim":
        if dims == 3:
            if size == 2:
                net = RSNAClassifier3D(is_dw=is_depth_wise, output_size=output_size)
                x = torch.rand(1,1,192,192,192)
                y = net(x,x)
                model = RSNAClassifier3D(is_dw=is_depth_wise, output_size=output_size)
            else:
                net = RSNAClassifierSingle3D(is_dw=is_depth_wise, output_size=output_size)
                x = torch.rand(1,1,192,192,192)
                y = net(x)
                model = RSNAClassifierSingle3D(is_dw=is_depth_wise, output_size=output_size)
        elif dims == 2:
            if size == 2:
                net = RSNAClassifierFake3D(is_dw=is_depth_wise, output_size=output_size)
                x = torch.rand(1,192,192,192)
                y = net(x,x)
                model = RSNAClassifierFake3D(is_dw=is_depth_wise, output_size=output_size)
            else:
                net = RSNAClassifierSingleFake3D(is_dw=is_depth_wise, output_size=output_size)
                x = torch.rand(1,192,192,192)
                y = net(x)
                model = RSNAClassifierSingleFake3D(is_dw=is_depth_wise, output_size=output_size)
    elif chosen_net == "vot":
        if dims == 2 and size == 1:
            net = RSNAClassifierSingleVoting2D(is_dw=is_depth_wise, output_size=output_size)
            x = torch.rand(1,1,192,192)
            y = net(x)
            model = RSNAClassifierSingleVoting2D(is_dw=is_depth_wise, output_size=output_size)
        else:
            print("Invalid chosen_net, returning default net RSNAClassifierSingle3D(is_dw=True, output_size=2)")
            return RSNAClassifierSingle3D(is_dw=True, output_size=output_size)
    elif chosen_net == "alt":
        if dims == 2 and size == 1:
            net = RSNAAlternativeClassifierSingle2D(is_dw=is_depth_wise, output_size=output_size)
            x = torch.rand(1,1,192,192)
            y = net(x)
            model = RSNAAlternativeClassifierSingle2D(is_dw=is_depth_wise, output_size=output_size)
        else:
            print("Invalid chosen_net, returning default net RSNAClassifierSingle3D(is_dw=True, output_size=2)")
            return RSNAClassifierSingle3D(is_dw=True, output_size=output_size)
    elif chosen_net == "eff":
        net = BinaryEfficientNet3D(output_size=output_size)
        x = torch.rand(1,1,192,192,192)
        y = net(x)
        model = BinaryEfficientNet3D(output_size=output_size)
    elif chosen_net == "vox":
        net = VoxCNN(output_size=output_size)
        x = torch.rand(2,1,192,192,192)
        y = net(x)
        model = VoxCNN(output_size=output_size)
    elif chosen_net == "res":
        net = RSNAResNet(output_size=output_size)
        x = torch.rand(2,1,192,192,192)
        y = net(x)
        model = RSNAResNet(output_size=output_size)
    elif chosen_net == "tun":
        net = TunAIResNet(output_size=output_size)
        x = torch.rand(1,1,192,192,192)
        y = net(x)
        model = TunAIResNet(output_size=output_size)
    elif chosen_net == "3du":
        net = UNet3d(in_channels=1, n_classes=1, n_channels=8)
        x = torch.rand(1,1,192,192,192)
        y = net(x)
        model = UNet3d(in_channels=1, n_classes=1, n_channels=8)
    else:
        print("Invalid chosen_net, returning default net RSNAClassifierSingle3D(is_dw=True, output_size=2)")
        return RSNAClassifierSingle3D(is_dw=True, output_size=output_size)
    
    return model

def get_model_from_path(modelfile):
    model_basename = os.path.basename(modelfile)
    net_name = model_basename.split("-")[0]
    if model_basename.split("-")[1] == "DW":
        net_is_dw = True
    else:
        net_is_dw = False
        
    if model_basename.split("-")[1] == "SO" or model_basename.split("-")[2] == "SO":
        output_size = 1
    else:
        output_size = 2
        
    if net_name == "RSNAClassifier3D":
        model = RSNAClassifier3D(is_dw=net_is_dw, output_size=output_size)
    elif net_name == "RSNAClassifierSingle3D":
        model = RSNAClassifierSingle3D(is_dw=net_is_dw, output_size=output_size)
    elif net_name == "RSNAClassifierFake3D":
        model = RSNAClassifierFake3D(is_dw=net_is_dw, output_size=output_size)
    elif net_name == "RSNAClassifierSingleFake3D":
        model = RSNAClassifierSingleFake3D(is_dw=net_is_dw, output_size=output_size)
    elif net_name == "RSNAClassifierSingleVoting2D":
        model = RSNAClassifierSingleVoting2D(is_dw=net_is_dw, output_size=output_size)
    elif net_name == "RSNAAlternativeClassifierSingle2D":
        model = RSNAAlternativeClassifierSingle2D(is_dw=net_is_dw, output_size=output_size)
    elif net_name == "BinaryEfficientNet3D":
        model = BinaryEfficientNet3D(output_size=output_size)
    elif net_name == "VoxCNN":
        model = VoxCNN(output_size=output_size)
    elif net_name == "RSNAResNet":
        model = RSNAResNet(output_size=output_size)
    elif net_name == "TunAIResNet":
        model = TunAIResNet(output_size=output_size)
    else:
        print("Invalid chosen_net, returning default net RSNAClassifierSingle3D(is_dw=True, output_size=2)")
        return RSNAClassifierSingle3D(is_dw=True, output_size=2)
    
    return model

def is_float(element: object):
    try:
        float(element)
        return True
    except ValueError:
        return False

def get_best_model(model_folder):
    best_acc = 0.0
    modelfile = ""
    info = {}
    for file in os.listdir(model_folder):
        if os.path.basename(file)[-4:] == ".pth":
            """
            auc_value = float(os.path.basename(file)[-9:-4])
            if auc_value > best_auc:
                best_auc = auc_value
                modelfile = os.path.basename(file)
                #print(auc_value)
            """
            acc_value = float(os.path.basename(file)[-18:-13])
            if acc_value > best_acc:
                best_acc = acc_value
                modelfile = os.path.basename(file)
        elif os.path.basename(file) == "training_info.txt":
            with open(f"{model_folder}/{file}", 'r') as f:
                #file.write(json.dumps(info))
                lines = f.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    if line != "{" and line != "}":
                        parts = line.split(": ")
                        if parts[1][0] == "[":
                            parts[1] = parts[1].replace("[", "").replace("]", "").replace('"', "").replace("'", "")
                            parts[1] = parts[1].split(", ")
                        elif parts[1].isnumeric():
                            parts[1] = int(parts[1])
                        elif is_float(parts[1]):
                            parts[1] = float(parts[1])
                        elif parts[1] == "true":
                            parts[1] = True
                        elif parts[1] == "false":
                            parts[1] = False
                        elif "e-" in parts[1] or "e+" in parts[1]:
                            if parts[1][0] == "e":
                                parts[1] = "1"+parts[1]
                            parts[1] = float(parts[1])
                        info[parts[0]] = parts[1]
            
    return modelfile, info
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert_to_onnx(model, device, modelfile, data_loader, size, is_target_included=True):
    logging.info("Converting {} to ONNX".format(modelfile))
    model.to(device)
    
    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint["model_state_dict"])

    if size == 2:
        for e, elem in enumerate(data_loader,1):
            if e == 1:
                (img_ids, images, labels) = elem
                with torch.no_grad():
                    if is_target_included:
                        X_1, targets = images[0].to(device), labels.to(device)
                        X_2, targets = images[1].to(device), labels.to(device)
                    else:
                        X_1 = images[0].to(device)
                        X_2 = images[1].to(device)

                    input_names = ['MRI Scan']
                    output_names = ['Tumor']
                    if "BinaryEfficientNet3D" in modelfile:
                        model.net.set_swish(memory_efficient=False)
                    torch.onnx.export(model, (X_1,X_2), f'{modelfile}.onnx', input_names=input_names, output_names=output_names)
    else:
        for e, elem in enumerate(data_loader,1):
            if e == 1:
                (img_ids, images, labels) = elem
                with torch.no_grad():
                    if is_target_included:
                        X, targets = images[0].to(device), labels.to(device)
                    else:
                        X = images[0].to(device)
                    modelname = model.__class__.__name__
                    if modelname == "RSNAClassifierSingleVoting2D" or modelname == "RSNAAlternativeClassifierSingle2D":
                        for i in range(X.shape[1]):
                            x_slice = torch.unsqueeze(X[:,i,:,:], dim=1)

                            if i == 1:
                                input_names = ['MRI Scan']
                                output_names = ['Tumor']
                                torch.onnx.export(model, x_slice, f'{modelfile}.onnx', input_names=input_names, output_names=output_names)
                    else:
                        input_names = ['MRI Scan']
                        output_names = ['Tumor']
                        if "BinaryEfficientNet3D" in modelfile:
                            model.net.set_swish(memory_efficient=False)
                        torch.onnx.export(model, X, f'{modelfile}.onnx', input_names=input_names, output_names=output_names)