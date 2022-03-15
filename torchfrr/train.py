from __future__ import division

import logging
import os
import os.path as osp
import signal
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose
from tqdm import tqdm

from utils.config import cfg_setup, get_cfg
cfg = get_cfg('train')
cfg_setup(cfg)
from datasets import get_datasets
from models import get_models
from transforms import get_transforms
from utils.io import log_init, resume_ckpt
from utils.lr_schedule import get_schedulers

# torch.autograd.set_detect_anomaly(True)
terminating = False


def main():
    log_init(cfg)
    logging.info(f'Configurations:  \n{OmegaConf.to_yaml(cfg)}')
    train(cfg)


def train(cfg):

    ds = get_datasets(cfg, ['train', 'val'])
    train_ds = ds['train']
    val_ds = ds['val']
    logging.info(train_ds)
    logging.info(val_ds)

    train_dataloader = DataLoader(train_ds, batch_size=cfg.TRAIN.BATCH_SIZE,
                                  shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, persistent_workers=(cfg.TRAIN.NUM_WORKERS > 0))
    val_dataloader = DataLoader(
        val_ds, num_workers=1, batch_size=cfg.VAL.BATCH_SIZE, shuffle=False)
    logging.info('Dataloader ready')

    read_trans = Compose(get_transforms(cfg.READ_TRANSFORMS))
    model = get_models(cfg.MODEL)
    loss_fn = Compose(get_transforms(cfg.TRAIN.LOSSES))
    train_metric_fn = Compose(get_transforms(cfg.TRAIN.METRICS))
    val_metric_fn = Compose(get_transforms(cfg.VAL.METRICS))
    logging.info('Model ready')

    model.cuda()
    logging.info('Cuda ready')

    opt = Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS)
    schedulers = get_schedulers(cfg, opt)
    logging.info('Opt ready')

    startepoch = 0
    global_step = 0
    if cfg.TRAIN.RESUME and os.listdir(osp.join(cfg.LOGDIR, 'ckpts')):
        lastepoch, global_step = resume_ckpt(
            cfg.CKPT_DIR, cfg.TRAIN.CKPT, model, opt)
        startepoch = lastepoch + 1
        logging.info(
            f'Resume training at epoch: {startepoch}, step: {global_step}')
    else:
        logging.info('Start new training')

    def handle_signal(signal_value, stack_frame):
        global terminating
        signame = signal.Signals(signal_value).name
        logging.info('Process {} got signal {}. Saving at epoch {}.'.format(
            os.getpid(), signame, epoch))
        torch.save({"model": model.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "opt": opt.state_dict()},
                   osp.join(cfg.LOGDIR, 'ckpts', f"{cfg.NAME}_{epoch:04d}_temp.pth"))
        terminating = True

    signal.signal(signal.SIGUSR1, handle_signal)

    for epoch in range(startepoch, cfg.TRAIN.MAX_EPOCH):

        logging.info(f"Epoch {epoch:04d} | lr: {opt.param_groups[0]['lr']}")
        model.train()

        for batch_id, batch in enumerate(train_dataloader):
            batch = defaultdict(dict, batch)
            batch.update({
                'epoch': epoch,
                'batch_id': batch_id,
                'global_step': global_step,
            })
            with torch.no_grad():
                batch = read_trans(batch)
            batch = model(batch)
            batch = loss_fn(batch)
            if not batch.get('invalid', False):
                opt.zero_grad(set_to_none=True)
                batch['losses']['total'].backward()
                opt.step()

            batch['metrics'].update(batch['losses'])
            with torch.no_grad():
                batch = train_metric_fn(batch)
            global_step += 1

        if epoch % cfg.TRAIN.CKPT_FREQ == 0 or terminating:
            logging.info(f"Checkpoint {cfg.NAME}_{epoch:04d}.pth")
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "opt": opt.state_dict()},
                       osp.join(cfg.LOGDIR, 'ckpts', f"{cfg.NAME}_{epoch:04d}.pth"))
            if terminating:
                logging.info(f'saved at epoch {epoch} before terminating')

        if epoch % cfg.VAL.FREQ == 0:
            validate(val_dataloader, read_trans, model,
                     val_metric_fn, epoch, loss_fn,)

        for s in schedulers:
            s.step()


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


if __name__ == '__main__':
    main()
