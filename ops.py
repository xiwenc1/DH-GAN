import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import common

def init_weights(modules):
    pass
            
    
class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out

class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out



class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicBlock(channel , channel // reduction, 1, 1, 0)
        self.c2 = BasicBlockSig(channel // reduction, channel , 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2

class SubbandPyramid(nn.Module):
    def __init__(self, in_channels):
        super(SubbandPyramid, self).__init__()
        self.attention_0 = CALayer(in_channels)
        self.attention_1 = CALayer(in_channels)
        self.attention_2 = CALayer(in_channels)
        self.attention_3 = CALayer(in_channels)
        
    def forward(self, x):
        x_low_l3, x_high_l3, x_high_l2, x_high_l1 = common.dwt_pyramid(x)
        
        x_low_l2 = common.iwt_init(torch.cat([self.attention_3(x_low_l3), x_high_l3],1))
        x_low_l1 = common.iwt_init(torch.cat([self.attention_2(x_low_l2), x_high_l2],1))
        org_l0 = common.iwt_init(torch.cat([self.attention_1(x_low_l1), x_high_l1],1))
        return self.attention_0(org_l0)


class Merge_Run(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_Run, self).__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.ReLU(inplace=True)
        )

        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        init_weights(self.modules)
        
    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class Merge_Run_dual(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_Run_dual, self).__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 4, 4),
            nn.ReLU(inplace=True)
        )

        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        init_weights(self.modules)
        
    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


class EResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
        )
        
        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x, inplace=True)
        return out
    
    