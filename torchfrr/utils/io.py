import datetime
import functools
import subprocess
from tensorboardX import SummaryWriter
import os
import os.path as osp
import torch
import logging
import utils
import cv2 as cv
import numpy as np
import lmdb
import yaml
import matplotlib.pyplot as plt
try:
    import rawpy ,exifread        
    from exifread.classes import IfdTag

except ImportError:
    print('raw related packages not installed')

def raw2rgbg(raw_array, cfa_pattern='rggb'):
    if cfa_pattern == 'rggb':
        rgbg = np.stack([raw_array[0::2, 0::2], raw_array[0::2, 1::2],
                         raw_array[1::2, 1::2], raw_array[1::2, 0::2]], axis=2)
    elif cfa_pattern == 'bggr':
        rgbg = np.stack([raw_array[1::2, 1::2], raw_array[0::2, 1::2],
                         raw_array[0::2, 0::2], raw_array[1::2, 0::2]], axis=2)

    return rgbg

def parse_exif(exif_dict):
    with open('utils/exiftag.yaml', 'r') as f:
        exiftag = yaml.load(f, Loader=yaml.FullLoader)
    for key in exif_dict.keys() & exiftag.keys():
        exif_dict[exiftag[key]] = exif_dict.pop(key)
    if 'Image WhiteLevel' not in exif_dict:
        exif_dict['Image WhiteLevel'] = IfdTag('Default Whitelevel', 0, 4, [
                                               16384], 0, 0)
    return exif_dict

def get_exif(path):
    with open(path, 'rb')as f:
        exif = parse_exif(exifread.process_file(f, details=True))
    return exif


def imread(path: str):
    """Read image unchanged
        Read rotated raw demosaic as RGBG - blacklevel """
    if path[-4:] == '.dng':
        exif = get_exif(path)
        black_level = exif['Image BlackLevel'].values
        white_level = exif["Image WhiteLevel"].values
        orientation_type = exif['Image Orientation'].values[0]
        cfa_pattern = exif['Image CFAPattern'].values
        white_level = 1023
        black_level = 64
        assert (np.array(black_level) == black_level).all()
        assert (np.array(white_level) == white_level).all()
        assert orientation_type == 1
        assert cfa_pattern == [0, 1, 1, 2]
        rotation = 0
        with rawpy.imread(path) as raw:
            img = np.rot90(np.clip((
                raw2rgbg(raw.raw_image_visible, 'rggb').astype(np.float32) - black_level) / (white_level - black_level) * 65535,
                0, 65535).astype(np.uint16), rotation)
    else:
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

