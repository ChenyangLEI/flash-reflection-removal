import pandas as pd
from torchvision.transforms.transforms import Compose
from torch.utils.data import ConcatDataset
from datasets.base import RangeDataset
from transforms.functional import imgs2float
from transforms.iotransforms import IdxFolderImgsRead
import os.path as osp
from datasets.registry import DATASETS


@DATASETS.register
class FoldersDataset(ConcatDataset):
    def __init__(self, root, subdirs, ls_name='trip.csv', transform=None, prefix='', **kwargs) -> None:
        dss = [RangeDataset(len(pd.read_csv(osp.join(root, subdir, ls_name))),
                            prefix + subdir, subdir=subdir, **kwargs)
               for subdir in subdirs]
        super().__init__(dss)
        if transform is None:
            transform = IdxFolderImgsRead(
                root, ls_name, ['ab_R', 'ab_T', 'fo'])
        self.transform = transform

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return self.transform(data)


@DATASETS.register
class Real2Dataset(FoldersDataset):
    def __init__(self, root, subdirs, ls_name='trip.csv', img_names=['ab_R', 'ab_T', 'fo'], transform=None, **kwargs) -> None:
        trans = [
            IdxFolderImgsRead(
                root, ls_name, img_names),
            imgs2float,
        ]
        if transform is not None:
            if isinstance(transform, list):
                trans.extend(transform)
            else:
                trans.append(transform)
        del kwargs['name']

        super().__init__(root, subdirs, ls_name, Compose(trans), **kwargs)
