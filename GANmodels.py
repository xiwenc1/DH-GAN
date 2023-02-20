import torch.nn as nn
import ops
import common
import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from models import ENET


# the architecture of the gan
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(2, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 16, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
        
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(16, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 2, 3, stride=1, padding=1),
        )
        
    
    def forward(self,x):
        x = self.conv_init(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_nonlinear(x)
        
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        return x
        

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16)
            #nn.Tanh(),
        )
        
        
        self.fc = nn.Sequential(            
            nn.Linear(16, 1)
            
        )
             

    def forward(self,x):
        x = self.conv_init(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_nonlinear(x)
        x = F.adaptive_avg_pool2d(x, (1,1)) #global pooling
        x =  x.view(1,-1)
        

        x = self.fc(x)

        return x
'''
class RECLoss(nn.Module):
    def __init__(self):
        super(RECLoss,self).__init__()
        self.Nx = 500
        self.Ny = 500
        self.z = z
        self.wavelength =wavelength
        self.deltaX = 4
        self.deltaY = 4
        self.prop = self.propagator(self.Nx,self.Ny,self.z,self.wavelength,self.deltaX,self.deltaY)
        self.prop = self.prop.cuda()

    def propagator(self,Nx,Ny,z,wavelength,deltaX,deltaY):
        k = 1/wavelength
        x = np.expand_dims(np.arange(np.ceil(-Nx/2),np.ceil(Nx/2),1)*(1/(Nx*deltaX)),axis=0)
        y = np.expand_dims(np.arange(np.ceil(-Ny/2),np.ceil(Ny/2),1)*(1/(Ny*deltaY)),axis=1)
        y_new = np.repeat(y,Nx,axis=1)
        x_new = np.repeat(x,Ny,axis=0)
        kp = np.sqrt(y_new**2+x_new**2)
        term=k**2-kp**2
        term=np.maximum(term,0) 
        phase = np.exp(1j*2*np.pi*z*np.sqrt(term))
        return torch.from_numpy(np.concatenate([np.real(phase)[np.newaxis,:,:,np.newaxis], np.imag(phase)[np.newaxis,:,:,np.newaxis]], axis = 3))
   

    def roll_n(self, X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def batch_fftshift2d(self, x):
        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim)//2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self.roll_n(real, axis=dim, n=n_shift)
            imag = self.roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    def batch_ifftshift2d(self,x):
        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim)//2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim)//2)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)
    
    def complex_mult(self, x, y):
        real_part = x[:,:,:,0]*y[:,:,:,0]-x[:,:,:,1]*y[:,:,:,1]
        real_part = real_part.unsqueeze(3)
        imag_part = x[:,:,:,0]*y[:,:,:,1]+x[:,:,:,1]*y[:,:,:,0]
        imag_part = imag_part.unsqueeze(3)
        return torch.cat((real_part, imag_part), 3)
    
    def TV(self,x,mask):
        batch_size = x.size()[0]
        mask_tensor = torch.zeros((x.size())).to(device)
        for i in range(batch_size):
            mask_tensor[i,:,:,0] = mask
            mask_tensor[i,:,:,1] = mask
        h_x = x.size()[2]
        w_x = x.size()[3]
        
        count_h = self._tensor_size(x[:,1:,:,:])
        count_w = self._tensor_size(x[:,:,1:,:])
        x = torch.mul(x,mask_tensor)
        amp = torch.sqrt(torch.pow(x[:,:,:,0],2)+torch.pow(x[:,:,:,1],2))
        phase = torch.atan2(x[:,:,:,0],x[:,:,:,1])
#         phase = (phase-torch.min(phase))/(torch.max(phase)-torch.min(phase))
#         h_tv = torch.pow(phase[:,1:,:]-phase[:,:h_x-1,:],2).sum() #gradient in horizontal axis
#         w_tv = torch.pow(phase[:,:,1:]-phase[:,:,:w_x-1],2).sum() #gradient in vertical axis
        
        
#         h_tv = torch.pow(x[:,1:,:,:]-x[:,:h_x-1,:,:],2).sum() #gradient in horizontal axis
#         w_tv = torch.pow(x[:,:,1:,:]-x[:,:,:w_x-1,:],2).sum() #gradient in vertical axis

#         return 0.005*2*(h_tv/count_h+w_tv/count_w)/batch_size #0.005 for cs prior
        return torch.sum(amp)/(batch_size*h_x*w_x)+torch.sum(phase)/(batch_size*h_x*w_x) #0.005 for cs prior
    
    def forward(self,x,y,mask,mylambda=1e-2):
        x = x.squeeze(2)
        y = y.squeeze(2)
        x = x.permute([0,2,3,1])
        y = y.permute([0,2,3,1])
    
        
        cEs = self.batch_fftshift2d(torch.fft(x,3,normalized=True))
        cEsp = self.complex_mult(cEs,self.prop)
        
        S = torch.ifft(self.batch_ifftshift2d(cEsp),3,normalized=True)
        Se = S[:,:,:,0]
#         print(self.TV(x,mask))
        loss = torch.mean(torch.abs(Se-torch.sqrt(y[:,:,:,0])))/2#+mylambda*self.TV(x,mask)#torch.mean(torch.abs(Se-y[:,:,:,0]))/2#
        
        return loss


    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
 



class BCELosswithLogits(nn.Module):
    def __init__(self, pos_weight=1, reduction='mean'):
        super(BCELosswithLogits, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, *], target: [N, *]
        logits = torch.sigmoid(logits)
        loss = - self.pos_weight * target * torch.log(logits) - \
               (1 - target) * torch.log(1 - logits)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss 
        
        
        
        
'''     
'''   
def propagator(Nx,Ny,z,wavelength,deltaX,deltaY):
        k = 1/wavelength
        x = np.expand_dims(np.arange(np.ceil(-Nx/2),np.ceil(Nx/2),1)*(1/(Nx*deltaX)),axis=0)
        y = np.expand_dims(np.arange(np.ceil(-Ny/2),np.ceil(Ny/2),1)*(1/(Ny*deltaY)),axis=1)
        y_new = np.repeat(y,Nx,axis=1)
        x_new = np.repeat(x,Ny,axis=0)
        kp = np.sqrt(y_new**2+x_new**2)
        term=k**2-kp**2
        term=np.maximum(term,0) 
        phase = np.exp(1j*2*np.pi*z*np.sqrt(term))
        return torch.from_numpy(np.concatenate([np.real(phase)[np.newaxis,:,:,np.newaxis], np.imag(phase)[np.newaxis,:,:,np.newaxis]], axis = 3))
   
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d( x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def complex_mult(x, y):
    real_part = x[:,:,:,0]*y[:,:,:,0]-x[:,:,:,1]*y[:,:,:,1]
    real_part = real_part.unsqueeze(3)
    imag_part = x[:,:,:,0]*y[:,:,:,1]+x[:,:,:,1]*y[:,:,:,0]
    imag_part = imag_part.unsqueeze(3)
    return torch.cat((real_part, imag_part), 3)

def forward_propogate(x):
    x = x.squeeze(2)
#     y = y.squeeze(2)
    x = x.permute([0,2,3,1])
#     y = y.permute([0,2,3,1])

    prop = propagator(Nx,Ny,z,wavelength,deltaX,deltaY).to(device, dtype=torch.float)
    cEs = batch_fftshift2d(torch.fft(x,3,normalized=True))
    cEsp =complex_mult(cEs,prop)

    S = torch.ifft(batch_ifftshift2d(cEsp),3,normalized=True)
    Se = S[:,:,:,0].unsqueeze(-1)
    Se = Se.permute([0,3,1,2])
    return Se
        
        

        
'''
class Generator_SPA(nn.Module):
    def __init__(self):
        super().__init__()
        n_feats = 32
        #self.enet = ENET(n_feats)
        # self.ca1 = ops.SubbandPyramid(32)
        # self.ca2 = ops.SubbandPyramid(32)
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(2, 32, 5, stride=1, padding=2),    
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            ops.SubbandPyramid(32),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            
            nn.BatchNorm2d(32),
            nn.ReLU(True),
           # ops.SubbandPyramid(32),
            
        )
        
        self.pooling_1 = nn.MaxPool2d(2)
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 16, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
        
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(16, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 2, 3, stride=1, padding=1),
        )
        
    
    def forward(self,x):
    
        p2d = (2,2,2,2)
        x =  F.pad(x, p2d, "constant", 0)
        
        #x = self.enet(x)      
        x = self.conv_init(x)
        
        x = x[:,:,2:-3,2:-3]
        x = self.pooling_1(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_nonlinear(x)
        
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        return x
        
        
        
        
