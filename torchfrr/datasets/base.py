from torch.utils.data import Dataset
from collections import defaultdict

from torchvision.transforms.transforms import Compose


class RangeDataset(Dataset):
    def __init__(self, size, name='dbg', transform=None, repeat=1, **kwargs):
        super().__init__()
        self.size = size
        self.repeat = repeat
        self.name = name
        if isinstance(transform, list):
            transform = Compose(transform)
        self.transform = transform
        self.kwargs = kwargs

    def __len__(self):
        return self.size * self.repeat

    def __getitem__(self, idx):
        data = defaultdict(dict)
        data.update({'dataset_name': self.name,
                     'dataset_size': self.size,
                     'idx': idx % self.size,
                     'nrep': idx // self.size})
        data.update(self.kwargs)
        if self.transform:
            data = self.transform(data)
        return data

    def repeat(self, n):
        self.repeat = n

class ListDataset(Dataset):
    def __init__(self, ls, transform=None):
        super().__init__()
        self.ls = ls
        self.transform = transform

    def __len__(self):
        return len(self.ls)

    def __getitem__(self, idx):
        data = self.ls[idx]

        if self.transform:
            data = self.transform(data)
        return data
