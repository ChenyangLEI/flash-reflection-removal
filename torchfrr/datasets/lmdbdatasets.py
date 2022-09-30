import os.path as osp

import pandas as pd
from torchvision.transforms.transforms import Compose
from transforms.functional import imgs2float
from transforms.iotransforms import IdxLmdbImgRead, RandLmdbImgRead
from transforms.transforms import InverseGamma, RenameImgs

from datasets.base import RangeDataset
from datasets.registry import DATASETS


@DATASETS.register
class SynLmdbDataset(RangeDataset):
    def __init__(self, afpath, rpath,
                 afls='afpair.csv', rls='abR.csv',
                 dataset_type='syn_synref', name='dbg', size=None,
                 reshuffle_each_iteration=True, transform=None, **kwargs):
        if size is None:
            size = len(pd.read_csv(osp.join(afpath, afls)))
        super().__init__(size, name, **kwargs)
        trans = [IdxLmdbImgRead(afpath, afls, ['ab_T', 'fo_T'], fmt='blosc')]
        if dataset_type == 'syn_synref':
            trans.extend([
                RandLmdbImgRead(rpath, rls, 'ab_R', fmt='blosc',
                                reshuffle_each_iteration=reshuffle_each_iteration),
            ])
        elif dataset_type == 'syn_corref':
            trans.extend([
                RandLmdbImgRead(rpath, rls, 'ab_R', fmt='img',
                                reshuffle_each_iteration=reshuffle_each_iteration),
                imgs2float,
                InverseGamma(('ab_R',)),
            ])
        trans.append(
            RenameImgs((
                ('fo_T', 'fo'),
            )),
        )
        if transform:
            trans.append(transform)
        self.transform = Compose(trans)
