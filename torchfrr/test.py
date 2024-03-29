from __future__ import division

import logging

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose

from tqdm import tqdm
import torch
# from train import validate
from utils.config import cfg_setup, get_cfg
cfg = get_cfg('test')
cfg_setup(cfg)
from datasets import get_datasets
from models import get_models
from transforms import get_transforms
from utils.io import log_init, resume_ckpt
from collections import defaultdict


def main():
    log_init(cfg)
    logging.info(f'Configurations:  \n{OmegaConf.to_yaml(cfg)}')
    test(cfg)

def validate(val_dataloader, read_trans, model, val_metric_fn, epoch, loss_fn=None):
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(val_dataloader, disable=None)):
            batch = defaultdict(dict, batch)
            batch.update({
                'epoch': epoch,
                'batch_id': batch_id,
            })
            batch = read_trans(batch)
            batch = model(batch)
            if loss_fn:
                batch = loss_fn(batch)
                batch['metrics'].update(batch['losses'])
            val_metric_fn(batch)

def test(cfg):

    test_ds = get_datasets(cfg, ['test'])['test']
    logging.info(test_ds)

    test_dataloader = DataLoader(
        test_ds, num_workers=1, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)
    logging.info('Dataloader ready')

    read_trans = Compose(get_transforms(cfg.TEST.READ_TRANSFORMS))
    model = get_models(cfg.MODEL)
    loss_fn = Compose(get_transforms(cfg.TRAIN.LOSSES))
    metric_fn = Compose(get_transforms(cfg.TEST.METRICS))
    logging.info('Model ready')

    model.cuda()
    logging.info('Cuda ready')

    epoch, _ = resume_ckpt(cfg.CKPT_DIR, cfg.TEST.CKPT, model)

    validate(test_dataloader, read_trans, model, metric_fn, epoch, loss_fn)


if __name__ == '__main__':
    main()
