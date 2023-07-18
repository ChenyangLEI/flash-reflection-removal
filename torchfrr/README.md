# Flash reflection removal pytorch implementation with alignment


## Prepare environment
```
conda env create -f environment.yaml  
conda activate frr
mim install mmcv-full
pip install mmflow
mim download mmflow --config pwcnet_ft_4x1_300k_sintel_final_384x768

# For training
pip install -r requirements_dev.txt
```

## Download Data and Pretrained models

Download test data and pretrained models from [onedrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/xjiangan_connect_ust_hk/EuM-fkWwAKlOjetolgSXRM4BEl1uai4mwLdiKAM_ka6_iA?e=HU63Gc)

For training, create a folder `dpt_weights` and download DPT pretrained weights [dpt_hybrid-midas-501f0c75.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt) there.


## Test

Test certain configuration and pretrained model.
For example:
```
GPU_ID=0
CONFIG=config/alignment/srgb2srgb_fo_dwpwc.py
EXP_NAME=srgb2srgb_fo_dwpwc8
TEST_NAME=align_test
python test.py --cfg $CONFIG --gpu $GPU_ID PHASE=hpc NAME=$EXP_NAME TEST.NAME=$TEST_NAME 
```

## Train
Train certain configuration.
For example:
```
GPU_ID=0
CONFIG=config/alignment/srgb2srgb_fo_dwpwc.py
EXP_NAME=srgb2srgb_fo_dwpwc233
python train.py --cfg $CONFIG --gpu $GPU_ID PHASE=hpc NAME=$EXP_NAME
```

## To Do
- [x] Release test code
- [x] Release dataset
- [x] Prepare paper and upload to arxiv
- [x] Release training code
- [ ] Release raw data processing code
