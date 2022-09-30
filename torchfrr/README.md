# Flash reflection removal pytorch implementation with alignment


## Prepare environment
```
conda env create -f environment.yml  
conda activate frr
mim install mmcv-full
pip install mmflow
mim download mmflow --config pwcnet_ft_4x1_300k_sintel_final_384x768
```

## Download Data and Pretrained models

Download test data and pretrained models from [onedrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/xjiangan_connect_ust_hk/EuM-fkWwAKlOjetolgSXRM4BEl1uai4mwLdiKAM_ka6_iA?e=HU63Gc)



## Test

Test certain configuration and pretrained model.
For example:
```
GPU_ID=0
CONFIG=config\alignment\srgb2srgb_fo_dwpwc.py
EXP_NAME=srgb2srgb_fo_dwpwc8
TEST_NAME=align_test
python test.py --cfg $CONFIG --gpu $GPU_ID PHASE=hpc NAME=$EXP_NAME TEST.NAME=$TEST_NAME 
```


