import argparse
import importlib
import os
import os.path as osp
import subprocess

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.misc import manual_seed, set_deterministic

cfg = None


def load_omgcfg(path: str):
    if path.endswith('.py'):
        path = path[:-3].replace('/', '.').replace('\\',".")
        cfg = importlib.import_module(path)._C.copy()
    elif path.endswith('.yaml'):
        cfg = OmegaConf.load(path)
        if 'INCLUDES' in cfg:
            includes = [load_omgcfg(x) for x in cfg.INCLUDES]
            cfg.merge_with(*includes)
    return cfg

def get_cfg(mode):
    # select configuration file and gpu according to argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='config/omega/defaults.py',
                        help="configuration file path")
    parser.add_argument("--gpu", default='', type=str, help="gpu id")
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.gpu == '':
        args.gpu = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
            "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

    # set configuration and gpu
    global cfg
    cfg = load_omgcfg(args.cfg)
    cfg.merge_with_dotlist(args.opts)
    cfg.CFG = args.cfg
    if args.gpu:
        cfg.GPU = args.gpu
    else:
        cfg.GPU = os.environ["CUDA_VISIBLE_DEVICES"][0]
    cfg.MODE = mode
    head, name = osp.split(osp.splitext(args.cfg)[0])
    phase = osp.split(head)[1]
    if not cfg.NAME:
        cfg.NAME = name
    if not cfg.PHASE:
        cfg.PHASE = phase
    OmegaConf.set_readonly(cfg, True)
    return cfg


def cfg_setup(cfg):
    manual_seed(cfg.SEED)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU
    print(f'CUDA {cfg.GPU} available:', torch.cuda.is_available())
    os.environ["OMP_NUM_THREADS"] = str(cfg.OMP_NUM_THREADS)

    if cfg.TRAIN.DETERMINISTIC:
        set_deterministic()
