from collections import defaultdict
import time
from torch.functional import Tensor
from torchvision.utils import make_grid
from utils.io import imdecode, imread, pkread, imwrite, open_lmdb
import numpy as np
import pickle
import pandas as pd
import os.path as osp
from utils.path import mkdir
import cv2 as cv
import os
from transforms.registry import TRANSFORMS
import utils
from utils.io import LMDBWriter, imtwrite
from utils.misc import AvgDict
import logging


@TRANSFORMS.register
class EpochImgsWrite:
    """ Write images to tbnail"""

    def __init__(self, root=None, prefix='train', img_names=None, save_freq=1, scale=1) -> None:
        self.root = root
        self.prefix = prefix
        self.img_names = img_names
        self.save_freq = save_freq
        self.step = int(1 / scale)

        self.epoch = None
        self.epoch_dir = None

    def __call__(self, data):
        batch_id = data['batch_id']
        epoch = data['epoch']
        if self.epoch != epoch:
            self.epoch = epoch
            self.epoch_dir = osp.join(self.root, f"{self.prefix}_{epoch:04d}")
            mkdir(self.epoch_dir)
        save_freq = data['save_freq'][0].item(
        ) if'save_freq' in data else self.save_freq

        if (batch_id % save_freq == 0) and (save_freq != 1 or data['nrep'] == 0):
            if self.img_names is None:
                self.img_names = tuple(data['imgs'].keys())
            for img_name in self.img_names:
                img: Tensor = data['imgs'][img_name].detach()
                if img.dim() > 3:
                    img = img[0]
                img = img[:, ::self.step, ::self.step]
                out_path = osp.join(self.epoch_dir,
                                    f"{batch_id:04d}_{data['dataset_name'][0]}_{data['idx'][0].item():04d}_{img_name}.jpg")
                imtwrite(out_path, img)
            # if imgs[3].shape[0] == 1:
            #     imgs[3] = imgs[3].expand(3, -1, -1)

        return data


@TRANSFORMS.register
class EpochMetricsLog:

    def __init__(self, root=None, prefix='train') -> None:
        self.root = root
        self.prefix = prefix
        self.tb_writer = utils.io.tb_writer
        self.metrics_dict = defaultdict(list)
        self.last_time = time.time()

        self.epoch = None
        self.epoch_dir = None

    def __del__(self):
        self.flush()

    def __call__(self, data):
        epoch = data['epoch']
        if self.epoch != epoch:
            self.flush()
            self.epoch = epoch
            self.epoch_dir = osp.join(self.root, f"{self.prefix}_{epoch:04d}")
            mkdir(self.epoch_dir)
        self.metrics_dict['dataset_name'].append(data['dataset_name'][0])
        self.metrics_dict['data_name'].append(
            f"{data['dataset_name'][0]}_{data['idx'][0].item():04d}")
        self.metrics_dict['batch_id'].append(str(data['batch_id']))
        for k, v in data['metrics'].items():
            self.metrics_dict[k].append(v.item())
        return data

    def flush(self):
        if self.epoch is None or len(self.metrics_dict) == 0:
            return
        df_metric = pd.DataFrame(
            self.metrics_dict)
        df_metric.set_index('data_name', inplace=True)
        df_metric.to_csv(
            osp.join(self.epoch_dir, f"metrics.csv"))
        avg_metric = df_metric.groupby(
            'dataset_name').mean(numeric_only=True)
        avg_metric.to_csv(osp.join(self.epoch_dir,
                                   f"summary.csv"))
        avg_metric = {'_'.join([ds_name, k]): v
                      for ds_name, ds_loss in avg_metric.iterrows()
                      for k, v in ds_loss.items()}
        for k, v in avg_metric.items():
            self.tb_writer.add_scalar(k, v, self.epoch)
        loss_message = ' | '.join(
            f"{k}: {v:.2f}" for k, v in avg_metric.items())
        current_time = time.time()
        logging.info("Epc:{:03d} | time:{:.3f}\n\t{}".format(
            self.epoch, current_time - self.last_time, loss_message))
        self.last_time = current_time

        self.metrics_dict.clear()


@TRANSFORMS.register
class StepMetricsLog:
    """ Log metrics in log freq"""

    def __init__(self, log_freq=100) -> None:
        self.log_freq = log_freq
        self.tb_writer = utils.io.tb_writer
        self.avg_dict = AvgDict()
        self.last_time = time.time()

    def __call__(self, data):
        global_step = data['global_step']
        self.avg_dict.update({k: v.item()
                              for k, v in data['metrics'].items()})

        if ((global_step % self.log_freq == 0) or
                ('log_freq' in data and global_step % data['log_freq'][0].item() == 0)):
            avg_metric = self.avg_dict.mean()
            self.avg_dict.clear()
            for k, v in avg_metric.items():
                self.tb_writer.add_scalar(
                    k, v, global_step)
            loss_message = ' | '.join(
                f"{k}: {v:.2f}" for k, v in avg_metric.items())
            current_time = time.time()
            logging.info("Epc:{:03d}-{:04d} | time:{:.3f}\n\t{}".format(
                data['epoch'], data['batch_id'], current_time - self.last_time, loss_message))
            self.last_time = current_time

        return data


@TRANSFORMS.register
class SixGridWrite:
    """ Write images to tbnail"""

    def __init__(self, save_freq=100, root=None, scale=0.5) -> None:
        self.root = root
        self.save_freq = save_freq
        self.step = int(1 / scale)
        self.epoch = None
        self.epoch_dir = None

    def __call__(self, data):
        batch_id = data['batch_id']
        epoch = data['epoch']
        if self.epoch != epoch:
            self.epoch = epoch
            self.epoch_dir = osp.join(self.root, f"{epoch:04d}")
            mkdir(self.epoch_dir)

        if ((batch_id % self.save_freq == 0) or
                ('save_freq' in data and batch_id % data['save_freq'][0].item() == 0)):
            imgs = [data['imgs'][x][0].detach() for x in (
                'ab', 'ab_T_pred', 'ab_T', 'fo', 'ab_R_pred', 'ab_R')]
            if imgs[3].shape[0] == 1:
                imgs[3] = imgs[3].expand(3, -1, -1)
            grid = make_grid(imgs, nrow=3)[:, ::self.step, ::self.step]
            out_path = osp.join(self.epoch_dir,
                                f"{data['dataset_name'][0]}_{data['idx'][0].item():04d}.jpg")
            imtwrite(out_path, grid)

        return data


class TbnailWrite:
    """ Write images to tbnail"""

    def __init__(self, root, outdir='tbn', scale=1 / 8) -> None:
        self.root = root
        self.scale = scale
        self.outdir = outdir
        mkdir(osp.join(root, outdir))

    def __call__(self, data):
        imgs = data['imgs']
        idx = data['idx']
        scale = self.scale
        for k, v in imgs.items():
            # print(k)
            # print(v.max())
            # print(v.shape)
            img = cv.resize((np.clip(v, 0, 1)**(1 / 2.2) *
                            255).astype(np.uint8), None, fx=scale, fy=scale)
            imwrite(osp.join(self.root, self.outdir,
                    f'{idx:03d}_{k}.jpg'), img)

        return data


class LmdbImgReader:
    def __init__(self, path, fmt='img') -> None:
        self.env = open_lmdb(path)
        self.fmt = fmt

    def __call__(self, name: str):
        with self.env.begin(write=False) as txn:
            imgc = txn.get(name.encode('ascii'))
        if self.fmt == 'img':
            img = imdecode(np.frombuffer(
                imgc, dtype=np.uint8))

        return img
        # return self.c


class IdxFolderImgsRead:
    def __init__(self, root, ls_name, img_names, fmt='lrgb') -> None:
        self.root = root
        self.ls_name = ls_name
        self.img_names = img_names
        self.fmt = fmt
        self.dfs = {}

    def __call__(self, data):
        idx = data['idx']
        subdir = data['subdir']
        if subdir not in self.dfs:
            self.dfs[subdir] = pd.read_csv(
                osp.join(self.root, subdir, self.ls_name))
        sample: pd.Series = self.dfs[subdir].iloc[idx]
        data['meta'][self.ls_name] = sample.to_dict()
        for imgname in self.img_names:
            name = sample[imgname]
            data['img_names'][imgname] = name
            data['imgs'][imgname] = imread(
                osp.join(self.root, subdir, self.fmt, name))
        return data


class IdxLmdbImgRead:
    def __init__(self, path, lsname, img_names, fmt='img') -> None:
        self.reader = LmdbImgReader(path, fmt=fmt)
        if not isinstance(img_names, list):
            img_names = [img_names]
        self.img_names = img_names
        self.ls = pd.read_csv(osp.join(path, lsname))[img_names]

    def __call__(self, data):
        sample = self.ls.iloc[data['idx']]
        for imgname in self.img_names:
            name = sample[imgname]
            data['img_names'][imgname] = name
            data['imgs'][imgname] = self.reader(name)

        return data


class RandLmdbImgRead:
    def __init__(self, path, lsname, imgname, fmt='img', reshuffle_each_iteration=True) -> None:
        self.reader = LmdbImgReader(path, fmt=fmt)
        self.ls = pd.read_csv(osp.join(path, lsname))[imgname]
        self.imgname = imgname
        self.idx = 0
        self.reshuffle_each_iteration = reshuffle_each_iteration
        self.shuffled = False

    def __call__(self, data):
        if self.idx == 0 and (not self.shuffled or self.reshuffle_each_iteration):
            self.ls = np.random.permutation(self.ls)
            self.shuffled = True
        if self.reshuffle_each_iteration:
            name = self.ls[self.idx]
        else:
            name = self.ls[data['idx']]
        data['img_names'][self.imgname] = name
        data['imgs'][self.imgname] = self.reader(name)
        self.idx += 1
        if self.idx == len(self.ls) or (not self.reshuffle_each_iteration and self.idx == data['dataset_size']):
            self.idx = 0

        return data

