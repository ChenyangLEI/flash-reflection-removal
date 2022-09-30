from omegaconf import OmegaConf

_C = OmegaConf.create()
_C.CFG = ''
_C.INCLUDES = []
# experiment phase, by default is the directory name of the config file
_C.PHASE = ''
# experiment name, by default is the config file name without extension
_C.NAME = ''
# experiment log directory (with subdirectory ckpts, tb, runs)
_C.LOGDIR = 'experiments/${PHASE}/${NAME}'
_C.VERBOSE = True
_C.DEBUG = False

_C.CKPT_DIR = '${LOGDIR}/ckpts'
_C.TRAINDIR = 'result/${PHASE}/${NAME}'
_C.MODE = 'test'
_C.OMP_NUM_THREADS = 4
_C.GPU = ''
_C.SEED = 2021

_C.DATASETS = OmegaConf.create()
_C.DATA_CFG = {}
_C.DATA_TRANSFORMS = {}
_C.READ_TRANSFORMS = []
_C.WRITE_TRANSFORMS = []

_C.TRAIN = OmegaConf.create()
_C.TRAIN.RESUME = False
_C.TRAIN.CKPT = -1
_C.TRAIN.DETERMINISTIC = False
_C.TRAIN.MAX_EPOCH = 151
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.BETAS = [0.9, 0.999]
_C.TRAIN.LR = 0.0001
_C.TRAIN.LOSS_LAYERS = 1
_C.TRAIN.NUM_WORKERS = 1
_C.TRAIN.LOG_FREQ = 100
_C.TRAIN.SAVE_FREQ = 100
_C.TRAIN.CKPT_FREQ = 50
_C.TRAIN.LOG_LOSS = False
_C.TRAIN.COS_LR = True
_C.TRAIN.LOSSES = []
_C.TRAIN.METRICS = []
_C.TRAIN.CONSISTENT_LOSS = []
_C.TRAIN.CONSISTENT_WEIGHT = 1.

_C.VAL = OmegaConf.create()
_C.VAL.FREQ = 1
_C.VAL.SAVE_FREQ = 1
_C.VAL.BATCH_SIZE = 1
_C.VAL.METRICS = []

_C.TEST = OmegaConf.create()
_C.TEST.NAME = 'test'
_C.TEST.CKPT = -1
_C.TEST.BATCH_SIZE = 1
_C.TEST.SAVE_FREQ = 1
_C.TEST.METRICS = []
_C.TEST.READ_TRANSFORMS = '${READ_TRANSFORMS}'

_C.EVAL = OmegaConf.create()
_C.EVAL.READ_TRANSFORMS= '${TEST.READ_TRANSFORMS}'
_C.EVAL.METRICS = []
_C.MODEL = [
    ['MultiUNet'],
]
