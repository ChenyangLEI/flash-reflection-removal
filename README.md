# Robust Reflection Removal with Reflection-free Flash-only Cues

### [Project Page]() | [Video]() | [Paper]() | [Data]()


Tensorflow implementation for: <br>
[Robust Reflection Removal with Reflection-free Flash-only Cues]()  
 [Chenyang Lei](https://chenyanglei.github.io/),
 [Qifeng Chen](https://cqf.io/) <br>
 HKUST
  
in CVPR 2021 

## To Do
- [ ] Prepare paper and upload to arxiv
- [ ] Make project page
- [ ] Release test code
- [ ] Release training code
- [ ] Release dataset
- [ ] Release raw data processing code

## TL;DR quickstart

To setup a conda environment, test on demo data:
```
conda env create -f environment.yml
conda activate flash-rr
python demo.py
```

## Setup

Python 3 dependencies:

* Tensorflow 1.13
* numpy
* imageio


## What is a RFC (Reflection-free Flash-only Cue)?

We propose a simple yet effective reflection-free cue for robust reflection removal from a pair of flash and ambient (no-flash) images. The reflection-free cue exploits a flash-only image obtained by subtracting the ambient image from the corresponding flash image in raw data space. The flash-only image is equivalent to an image taken in a dark environment with only a flash on.



## Citation

If you find our work useful for your research, please consider citing the following papers :)


If you are also interested in the other reflection removal methods, please refer to [this work]().


## Contact

Please contact me if there is any question (Chenyang Lei, leichenyang7@gmail.com)


## License

TBD
