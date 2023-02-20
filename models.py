import torch.nn as nn
import ops
import common
import torch

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = ops.Merge_Run_dual(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels, out_channels)
        self.r3 = ops.EResidualBlock(in_channels, out_channels)
        self.ca = ops.SubbandPyramid(in_channels)

    def forward(self, x):
        
        r1 = self.r1(x)            
        r2 = self.r2(r1)       
        r3 = self.r3(r2)
        out = self.ca(r3)
        
        return out        


class ENET(nn.Module):
    def __init__(self, n_feats):
        super(ENET, self).__init__()
        
        self.body = nn.Sequential(
            nn.Conv2d(2, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        self.ca = ops.SubbandPyramid(n_feats)
        self.tail = nn.Conv2d(n_feats, 2, 3, 1, 1)
    def forward(self,x):
        out = self.body(x)
        y= out.shape
        out = self.ca(out)
        return self.tail(out)

class SPANET(nn.Module):
    def __init__(self):
        super(SPANET, self).__init__()
        
        n_feats = 64
        kernel_size = 3
        
        self.enet = ENET(n_feats)
        
        self.head = ops.BasicBlock(6, n_feats, kernel_size, 1, 1)

        self.blc3 = nn.Sequential(
            Block(n_feats, n_feats),
            Block(n_feats, n_feats),
        )
        
        
        self.blc2 = nn.Sequential(
            Block(n_feats, n_feats),
            Block(n_feats, n_feats),
        )
        
        self.blc1 = nn.Sequential(
            Block(n_feats, n_feats),
            Block(n_feats, n_feats),
            Block(n_feats, n_feats),
            Block(n_feats, n_feats),
        )
        
        self.blc0 = nn.Sequential(
            Block(n_feats, n_feats),
            Block(n_feats, n_feats),
            Block(n_feats, n_feats),
            Block(n_feats, n_feats),
        )
        
        

        self.tail = nn.Conv2d(n_feats, 3, kernel_size, 1, 1, 1)

    def forward(self, x):
        
        est = self.enet(x)
        
        h = self.head(torch.cat([x,est],1))
        
        x_low_l3, x_high_l3, x_high_l2, x_high_l1 = common.dwt_pyramid(h)
        
        x_low_l3_pro = self.blc3(x_low_l3)+x_low_l3
        
        x_low_l2 = common.iwt_init(torch.cat([x_low_l3_pro, x_high_l3],1))
        
        x_low_l2_pro = self.blc2(x_low_l2)+x_low_l2
        
        x_low_l1 = common.iwt_init(torch.cat([x_low_l2_pro, x_high_l2],1))
        
        x_low_l1_pro = self.blc1(x_low_l1)+x_low_l1
        
        x_org_l0 = common.iwt_init(torch.cat([x_low_l1_pro, x_high_l1],1))
        
        x_org_l0_pro = self.blc0(x_org_l0)+x_org_l0

        res = self.tail(x_org_l0_pro)

        return res + x  
        
        

