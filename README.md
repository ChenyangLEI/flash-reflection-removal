# Robust Reflection Removal with Reflection-free Flash-only Cues (RFC)
<img src='example.jpg'/>

### [Paper](https://arxiv.org/pdf/2103.04273.pdf) | To be released: [Project Page]() | [Video]() | [Data]()


Tensorflow implementation for: <br>
[Robust Reflection Removal with Reflection-free Flash-only Cues]()  
 [Chenyang Lei](https://chenyanglei.github.io/),
 [Qifeng Chen](https://cqf.io/) <br>
 HKUST
  
in CVPR 2021 

## To Do
- [x] Release test code
- [x] Prepare paper and upload to arxiv
- [ ] Make project page
- [ ] Release training code
- [ ] Release dataset
- [ ] Release raw data processing code

## TL;DR quickstart

To setup a conda environment, test on demo data:
```
conda env create -f environment.yml
conda activate flashrr-rfc
bash download.sh
python test.py
```

## Setup

### Environment
This code is based on tensorflow. It has been tested on Ubuntu 18.04 LTS.

Anaconda is recommended: [Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-18-04)
| [Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)

After installing Anaconda, you can setup the environment simply by

```
conda env create -f environment.yml
```

### Download checkpoint and VGG model

Download the ckpt and VGG model by
```
bash download.sh
```



## What is a RFC (Reflection-free Flash-only Cue)?

We propose a simple yet effective reflection-free cue for robust reflection removal from a pair of flash and ambient (no-flash) images. The reflection-free cue exploits a flash-only image obtained by subtracting the ambient image from the corresponding flash image in raw data space. The flash-only image is equivalent to an image taken in a dark environment with only a flash on.



## Citation

If you find our work useful for your research, please consider citing the following papers :)

```
@misc{lei2021robust,
      title={Robust Reflection Removal with Reflection-free Flash-only Cues}, 
      author={Chenyang Lei and Qifeng Chen},
      year={2021},
      eprint={2103.04273},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

or 

```
@InProceedings{Lei_2021_RFC,
     title={Robust Reflection Removal with Reflection-free Flash-only Cues}, 
     author={Chenyang Lei and Qifeng Chen},
     booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
     year = {2021}
}
```
If you are also interested in the polarization reflection removal, please refer to [this work](https://github.com/ChenyangLEI/polarization-reflection-removal).


## Contact

Please contact me if there is any question (Chenyang Lei, leichenyang7@gmail.com)


## License

TBD
