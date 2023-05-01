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

