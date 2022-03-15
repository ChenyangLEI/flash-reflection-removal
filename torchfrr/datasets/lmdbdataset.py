import pandas as pd
from datasets.base import RangeDataset
from transforms.functional import imgs2float
from transforms.iotransforms import IdxLmdbImgRead
import os.path as osp
from datasets.registry import DATASETS


@DATASETS.register
class SrgbLmdbDataset(RangeDataset):
    def __init__(self, path, ls, name='dbg', size=None,
                 transform=None, **kwargs):
        path = osp.join(path, 'lmdb')
        if size is None:
            size = len(pd.read_csv(osp.join(path, ls)))

        trans = [IdxLmdbImgRead(
            path, ls, ['ab', 'ab_T', 'ab_R', 'fo']), imgs2float, ]
        if transform:
            if isinstance(transform, list):
                trans.extend(transform)
            else:
                trans.append(transform)
        super().__init__(size, name, transform=trans, ** kwargs)