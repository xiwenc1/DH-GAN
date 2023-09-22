# DH-GAN
The code for the paper: DH-GAN: A Physics-driven Untrained Generative Adversarial Network for 3D Microscopic Imaging using Digital Holography

**The paper has been accpeted by Optics Express and will be online soon.** 
For current version please view: https://preprints.opticaopen.org/articles/preprint/DH-GAN_A_Physics-driven_Untrained_Generative_Adversarial_Network_for_Holographic_Imaging/22009556 
## Contact
- Xiwen Chen (xiwenc@g.clemson.edu)
- Hao Wang (hao9@g.clemson.edu)

## Code
- Runnable code is in the notebook file ``` release_version.ipynb ```
- Environment is listed in ``` requirements.txt ```.
**Note the pytorch version is 1.6.0**
- ```mylabmda``` is the weight for masking loss. Please start with a small value (from 0).

- A saved running result is in ``` release_version.html ```


## Framework
![](https://github.com/XiwenChen-Clemson/DH-GAN/blob/main/figs/framework.jpg)


## Update loss function compatible for torch >1.6 (torch.fft has been substantially updated in nee version)
```
class RECLoss(nn.Module):
    def __init__(self):
        super(RECLoss,self).__init__()
        self.Nx = 500
        self.Ny = 500
        
        self.wavelength = wavelength
        self.deltaX = deltaX
        self.deltaY = deltaY
        # self.z = z
        # self.prop = self.propagator(self.Nx,self.Ny,self.z,self.wavelength,self.deltaX,self.deltaY)
        # self.prop = self.prop.cuda()

    def propagator(self,Nx,Ny,z,wavelength,deltaX,deltaY):
        k = 1/wavelength
        # x = np.expand_dims(np.arange(np.ceil(-Nx/2),np.ceil(Nx/2),1)*(1/(Nx*deltaX)),axis=0)
        x =torch.unsqueeze(torch.arange(\
                                        torch.ceil(-torch.tensor(Nx)/2),torch.ceil(torch.tensor(Nx)/2),1)*(1/(Nx*deltaX)),dim=0)
        # y = np.expand_dims(np.arange(np.ceil(-Ny/2),np.ceil(Ny/2),1)*(1/(Ny*deltaY)),axis=1)
        y = torch.unsqueeze(torch.arange(torch.ceil(-torch.tensor(Ny)/2),torch.ceil(torch.tensor(Ny)/2),1)*(1/(Ny*deltaY)),dim=1)
        
        # print(x.shape)
        # print(y.shape)
        # y_new = np.repeat(y,Nx,axis=1)
        y_new = y.repeat(1, Nx)
        # x_new = np.repeat(x,Ny,axis=0)
        x_new = x.repeat(Ny,1)
        # print(y_new.shape)
        # print(x_new.shape)
        
        kp = torch.sqrt(y_new**2+x_new**2)
        term=k**2-kp**2
        term=np.maximum(term,0) 
        phase = torch.exp(1j*2*torch.pi*z*np.sqrt(term))
        # return torch.from_numpy(np.concatenate([np.real(phase)[np.newaxis,:,:,np.newaxis], np.imag(phase)[np.newaxis,:,:,np.newaxis]], axis = 3))
        return torch.cat([torch.real(phase).reshape(1,phase.shape[0],phase.shape[1],1), torch.imag(phase).reshape(1,phase.shape[0],phase.shape[1],1)], dim = 3)

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
    
    def TV(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,1:,:,:])
        count_w = self._tensor_size(x[:,:,1:,:])
        h_tv = torch.pow((x[:,1:,:,:]-x[:,:h_x-1,:,:]),2).sum() #gradient in horizontal axis
        w_tv = torch.pow((x[:,:,1:,:]-x[:,:,:w_x-1,:]),2).sum() #gradient in vertical axis
        return 0.01*2*(h_tv/count_h+w_tv/count_w)/batch_size
    
    def forward(self,x,y,z= torch.tensor(5000.) ):
        x = x.squeeze(2)
        y = y.squeeze(2)
        x = x.permute([0,2,3,1])
        y = y.permute([0,2,3,1])
        
        self.z = z.squeeze().cpu()
        self.z = self.z.cpu()
        self.prop = self.propagator(self.Nx,self.Ny,self.z,self.wavelength,self.deltaX,self.deltaY)
        self.prop = self.prop.cuda()
        
        temp_x=torch.view_as_complex(x.contiguous())
          
       
        
        # cEs = self.batch_fftshift2d(torch.fft(x,3,normalized=True))
        
        cEs = self.batch_fftshift2d(torch.view_as_real (torch.fft.fftn(temp_x, dim=(0,1,2), norm="ortho")))
        
        cEsp = self.complex_mult(cEs,self.prop)
        
        # S = torch.ifft(self.batch_ifftshift2d(cEsp),3,normalized=True)
        
        temp = torch.view_as_complex(self.batch_ifftshift2d(cEsp).contiguous())
        S = torch.view_as_real(torch.fft.ifftn(temp, dim=(0,1,2), norm="ortho") )
        
        
        Se = S[:,:,:,0]
        
        loss = torch.mean(torch.abs(Se-torch.sqrt(y[:,:,:,0])))/2#torch.mean(torch.abs(Se-y[:,:,:,0]))/2#
        return loss


    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



```
## Cite 
If you find the code helpful for you, please kindly cite these papers
```
@article{chen2023dh,
  title={DH-GAN: a physics-driven untrained generative adversarial network for holographic imaging},
  author={Chen, Xiwen and Wang, Hao and Razi, Abolfazl and Kozicki, Michael and Mann, Christopher},
  journal={Optics Express},
  volume={31},
  number={6},
  pages={10114--10135},
  year={2023},
  publisher={Optica Publishing Group}
}
```
AND
```
@article{li2020deep,
  title={Deep DIH: single-shot digital in-line holography reconstruction by deep learning},
  author={Li, Huayu and Chen, Xiwen and Chi, Zaoyi and Mann, Christopher and Razi, Abolfazl},
  journal={IEEE Access},
  volume={8},
  pages={202648--202659},
  year={2020},
  publisher={IEEE}
}
```



## Some results
### Simulated hologram
![Simluated Holo](https://github.com/XiwenChen-Clemson/DH-GAN/blob/main/figs/holo.bmp)

### Reconstructed results
![Simluated Holo](https://github.com/XiwenChen-Clemson/DH-GAN/blob/main/figs/final_results_2.jpg)

### Results on real data
![](https://github.com/XiwenChen-Clemson/DH-GAN/blob/main/figs/DH_rec_1.png)

### Denosing
![](https://github.com/XiwenChen-Clemson/DH-GAN/blob/main/figs/NOISE_NEW.jpg)

**For more experiments with different structures, please ref the paper..**

