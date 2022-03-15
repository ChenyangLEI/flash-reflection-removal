

import os
import sys
from argparse import ArgumentParser


from functools import partial
from glob import glob
from os import path as osp
import pathlib
path=str(pathlib.Path(__file__).parent.parent.resolve().absolute())
sys.path.append(path)

import cv2 as cv
import numpy as np
import pandas as pd
from datasets.base import ListDataset
from utils.io import ds2folderlmdb, imencode, imread


def path2byte_meta(path, scale=1):

    img = imread(path)
    if scale == 1:
        with open(path, 'rb') as f:
            imgc = f.read()
    else:
        img = cv.resize(img, None, fx=scale, fy=scale)
        imgc = imencode(osp.splitext(path)[1], img)
    img = img / np.iinfo(img.dtype).max
    h, w, c = img.shape
    k = int(img.size * 0.99)
    kth = np.partition(img, k, axis=None)[k]
    return {
        'name': osp.basename(path),
        'payload': imgc,
        'h': h,
        'w': w,
        'mean': img.mean(),
        'max': img.max(),
        '1perc': kth
    }


def folder2lmdb(in_path='rawfr/refreal/jpg', db_path='rawfr/refreal/lmdb', scale=1):
    ls = glob(osp.join(in_path, '*'))
    imls2lmdb(ls, db_path, scale=scale)


def imls2lmdb(imls, db_path='data/rawfr/refreal/lmdb', scale=1):
    ls = sorted(imls)
    ds = ListDataset(ls, transform=partial(path2byte_meta, scale=scale))
    ds2folderlmdb(ds, db_path, num_workers=4)


def create_slmdb(root='data/real_world', mode='test', scale=1):
    root = osp.join(root, mode)
    in_path = osp.join(root, 'others')
    db_path = osp.join(root, 'lmdb')
    folder2lmdb(in_path, db_path, scale)
    df = pd.read_csv(osp.join(db_path, 'meta.csv'))
    names = np.array(sorted(df['name'])).reshape(-1, 5)
    dfg = pd.DataFrame(data=names, columns=['ab', 'ab_R', 'ab_T', 'f', 'fo'])
    dfg.to_csv(osp.join(db_path, f'{mode}.csv'), index=False)

if __name__ == '__main__':
    parser=ArgumentParser()
    parser.add_argument('dataroot',type=str,default='../data')
    args=parser.parse_args()
    root=args.dataroot

    for mode in ('train','val', 'test'):
        create_slmdb(osp.join(root,'real_world'), mode, scale=1)

    for ds in ('synthetic/with_syn_reflection','synthetic/with_corrn_reflection'):
        for mode in ('train', 'test'):
            create_slmdb(osp.join(root,ds), mode, scale=1)
