import os
import random
import numpy as np
import torch


def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_deterministic():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    
class AvgDict(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def clear(self):
        self.data.clear()
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError(
                        "invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError(
                        "invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}

def zip_broadcast(*ls):
    maxlen = max(len(x) for x in ls if not isinstance(ls, str))
    ls = (itertools.repeat(x, maxlen) if isinstance(x, str) else x for x in ls)
    return list(zip(*ls))

