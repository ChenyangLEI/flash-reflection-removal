
import logging
from collections import defaultdict

from torch.utils.data.dataset import ConcatDataset
from torchvision.transforms.transforms import Compose
from transforms import get_transforms

import datasets.folderdatasets
import datasets.lmdbdatasets
from datasets.registry import DATASETS


def build_dataset(name, cfg, trans=[]):
    logging.info(f'building dataset: {name}')

    if 'transforms' in cfg.keys():
        trans = get_transforms(cfg.transforms) + trans
    ds = DATASETS[cfg.dataset_type](
        name=name, transform=Compose(trans), **cfg.dataset_args)
    return ds


def get_datasets(cfg, phases):
    ds_dict = defaultdict(list)
    for name, dscfg in cfg.DATASETS.items():
        if dscfg.phase in phases:
            ds_dict[dscfg.phase].append(build_dataset(name, dscfg))
    ds_dict = {k: ConcatDataset(v) for k, v in ds_dict.items()}
    return ds_dict
