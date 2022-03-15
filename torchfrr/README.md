# Flash reflection removal pytorch implementation


## Prepare environment
```
conda env create -f environment.yml  
conda activate frr
```

## Download Data

The same data as the tensorflow implementation

## Prepare Data

Put data in lmdb database

```
DATA_PATH=../data
python scripts/create_lmdb.py $DATA_PATH
```

## Train

```
GPU_ID=0
python train.py --gpu $GPU_ID --cfg config/srgb/srgb2srgb_fo.py
```

## Test

```
GPU_ID=0
python test.py --gpu $GPU_ID --cfg config/srgb/srgb2srgb_fo.py
```


