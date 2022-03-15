import datetime
import functools
import json
import logging
import os
import os.path as osp
import pickle
import subprocess
from collections import defaultdict

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2 as cv
import lmdb
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import utils
from utils.misc import identity


def get_ckpt(ckpt_dir, epoch):
    ckpts = os.listdir(ckpt_dir)
    if not os.listdir(ckpt_dir):
        logging.error('Empty checkpoint directory')
        return

    if epoch < 0:
        ckpt = sorted(ckpts)[-1]
    else:
        name = utils.config.cfg.NAME
        ckpt = f'{name}_{epoch:04d}.pth'
    return ckpt


def resume_ckpt(ckpt_dir, epoch, model, opt=None):
    ckpt = get_ckpt(ckpt_dir, epoch)
    logging.info(f"Resume {ckpt}")
    state_dict = torch.load(osp.join(ckpt_dir, ckpt), map_location='cuda')
    model.load_state_dict(state_dict['model'])
    if opt is not None:
        if 'opt' in state_dict and state_dict['opt'] is not None:
            opt.load_state_dict(state_dict['opt'])
    epoch = state_dict['epoch']
    global_step = state_dict['global_step']
    return epoch, global_step


@functools.lru_cache
# prevent open database twice which may corrupt the database
def open_lmdb(db_path, read_only=True):
    if read_only:
        return lmdb.open(db_path, subdir=osp.isdir(db_path),
                         readonly=True, lock=False,
                         readahead=False, meminit=False)
    else:
        return lmdb.open(db_path, subdir=osp.isdir(db_path),
                         map_size=2**37, readonly=False,
                         meminit=False, map_async=True)


class LMDBWriter:
    def __init__(self, db_path, append=None, freq_w=64) -> None:
        if osp.exists(db_path):
            self.append = False
            print(f'exist lmdb: {db_path}')
        else:
            os.makedirs(db_path)
            self.append = True
        if append is not None:
            self.append = append
        self.db_path = db_path
        self.db = open_lmdb(db_path, read_only=False)
        self.txn = self.db.begin(write=True)
        self.idx = 0
        self.freq_w = freq_w

    def __call__(self, data):
        k, v = data
        success = self.txn.put(k, v, append=self.append)
        if not success:
            print(f'fail to write lmdb: {k}')
        if self.idx % self.freq_w == 0:
            self.commit()

        self.idx += 1

        return data

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.txn.commit()
        self.db.sync()
        self.db.close()

    def commit(self):
        self.txn.commit()
        self.txn = self.db.begin(write=True)


def scan_ds(ds, n=None, num_workers=1):
    dataloader = DataLoader(
        ds, num_workers=num_workers, batch_size=None, collate_fn=identity)
    for i, x in enumerate(tqdm(dataloader)):
        if n and i >= n:
            break
    return x


def ds2folderlmdb(ds, db_path, num_workers=1):
    """ Require data.name: str, data.payload:bytes"""

    dataloader = DataLoader(
        ds, num_workers=num_workers, batch_size=None, collate_fn=lambda x: x)
    df = defaultdict(list)

    with LMDBWriter(db_path, append=False) as writer:
        for i, data in enumerate(tqdm(dataloader)):
            name: str = data['name']
            payload = data.pop('payload')
            writer((name.encode('ascii'), payload))
            for col in data:
                df[col].append(data[col])

    df = pd.DataFrame(df)
    df.to_csv(osp.join(db_path, 'meta.csv'), index=False)
    df.to_markdown(osp.join(db_path, 'meta.md'), index=False, floatfmt='.3f')


def ds2namelmdb(ds, db_path, num_workers=1):
    dataloader = DataLoader(
        ds, num_workers=num_workers, batch_size=None, collate_fn=lambda x: x)
    with LMDBWriter(db_path, append=False) as writer:
        for i, data in enumerate(tqdm(dataloader)):
            img_names = data['img_names']
            imgs = data['imgs']
            for k in img_names:
                writer((img_names[k].encode('ascii'),
                        pickle.dumps(imgs[k], protocol=-1)))


def ds2idxlmdb(ds, db_path, num_workers=1):
    dataloader = DataLoader(
        ds, num_workers=num_workers, batch_size=None, collate_fn=lambda x: x)
    with LMDBWriter(db_path) as writer:
        for i, data in enumerate(tqdm(dataloader)):
            writer((i.to_bytes(4, 'big'), pickle.dumps(data, protocol=-1)))


def tbnail_md(root, name, n, columns):
    lines = ['|'.join(['<img src="{}" width=320>'.format(
        f'{name}/{idx:03d}_{c}.jpg') for c in columns]) for idx in range(n)]
    with open(osp.join(root, name + '.md'), 'w') as f:
        f.write('|'.join(columns) + '  \n')
        f.write('|'.join(['-' * 3] * len(columns)) + '  \n')
        f.write('  \n'.join(lines))


def txtwrite(path, s):
    with open(path, 'w') as f:
        f.write(s)


def pkread(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pkwrite(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def jsread(path):
    with open(path, 'r') as f:
        a = json.load(f)
    return a


def jswrite(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def imtwrite(path, img):
    """Write tensor image in C, H, W and range [0,1]"""
    from mmcv import flow2rgb

    # print(img.shape)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    c = img_np.shape[-1]
    if c == 2:
        img_np = flow2rgb(img_np)
    elif c == 1:
        fig, ax = plt.subplots()
        fig.tight_layout(pad=0)
        pos = ax.imshow(img_np, cmap='turbo')
        ax.set_title(f'mean_{img_np.mean():.3f}')
        fig.colorbar(pos, ax=ax)
        plt.savefig(path)
        plt.close(fig)
        return
    img = np.clip(img_np * 255, 0,
                  255).astype(np.uint8)
    imwrite(path, img)





tb_writer = None


def log_init(cfg):
    time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.isdir(cfg.CKPT_DIR):
        os.makedirs(cfg.CKPT_DIR)
    if not os.path.isdir(osp.join(cfg.LOGDIR, 'eval')):
        os.makedirs(osp.join(cfg.LOGDIR, 'eval'))
    run_dir = osp.join(cfg.LOGDIR, 'runs')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  datefmt='%y/%m/%d %H:%M:%S')

    if cfg.VERBOSE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(
        os.path.join(run_dir, f'{cfg.MODE}_{time_stamp}.log'), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    git_info = subprocess.run(
        ['git', 'log', '-s', '-1'], stdout=subprocess.PIPE, text=True)
    logging.info(f'Last commit: \n{git_info.stdout}')
    global tb_writer
    tb_writer = SummaryWriter(osp.join(cfg.LOGDIR, 'tb'))
    return tb_writer



def mdwrite(path: str, df: pd.DataFrame):
    df.to_csv(path, sep=' ', line_terminator="  \n", index=False, header=False)



def imread(path: str):
    """Read image unchanged
        Read rotated raw demosaic as RGBG - blacklevel """
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    if img is None:
        print('fail to read:', path, flush=True)
    elif len(img.shape) == 2:
        img = img
    elif img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
    else:
        assert img.shape[2] in [3, 4]
    return img


def imdecode(imgc, flag=cv.IMREAD_UNCHANGED):
    img = cv.imdecode(imgc, flag)
    if img is None:
        print('fail to decode', flush=True)
    elif len(img.shape) == 2:
        img = img
    elif img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
    else:
        assert img.shape[2] in [3, 4]
    return img



def imwrite(path: str, img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        cv.imwrite(path, img)
    elif img.shape[2] == 3:
        cv.imwrite(path, cv.cvtColor(img, cv.COLOR_RGB2BGR))
    elif img.shape[2] == 4:
        cv.imwrite(path, cv.cvtColor(img, cv.COLOR_RGBA2BGRA))
    else:
        assert img.shape[2] in [3, 4], img.shape


def imencode(ext, img):
    if len(img.shape) == 2:
        img = img
    elif img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    elif img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_RGBA2BGRA)
    else:
        assert img.shape[2] in [3, 4], img.shape

    result, imgc = cv.imencode(ext, img)
    if not result:
        print(f'fail to encode, img shape: {img.shape}')

    return imgc
