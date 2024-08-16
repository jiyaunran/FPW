import math
import torch
from torch import nn
from torch.nn.functional import relu, sigmoid

class StegaStampDecoder(nn.Module):
    def __init__(self, channels=1, image_in=False):
        super(StegaStampDecoder, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        num_of_layers = 17
        if image_in:
            layers.append(nn.Conv2d(in_channels=6*channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        else:
            layers.append(nn.Conv2d(in_channels=2*channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out

class Block(nn.Module):
    def __init__(self, in_dim, dim):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_dim, dim,    3, stride=1, padding=1)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        
        return x
        
class ResBlock(nn.Module):
    def __init__(self, in_dim, dim):
        super(ResBlock, self).__init__()
        if in_dim != dim:
            self.res_conv = nn.Conv2d(in_dim, dim, 1)
        else:
            self.res_conv = nn.Identity()
        
        self.block = Block(in_dim, dim)
    
    def forward(self, x):
        return self.block(x) + self.res_conv(x)
        
class PoisonGenerator(nn.Module):
    def __init__(self, in_dim=512, dim=64):
        super(PoisonGenerator, self).__init__()
        
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())
            
        #self.en_block1 = Residual(Block(dim, dim*2))    
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, dim, 7, stride=2, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            ResBlock(dim, dim*2), 
            nn.MaxPool2d(2, stride=2),
            ResBlock(dim*2, dim*4),
            nn.MaxPool2d(2, stride=2),
            ResBlock(dim*4, dim*8),
            nn.MaxPool2d(2, stride=2) # b, 512, 1, 1
        )
            
    def forward(self, x):
        y = self.encoder(x).view(x.shape[0], -1) # b, 512
        y = self.l1(y)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y