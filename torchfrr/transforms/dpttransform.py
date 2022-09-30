try:
    from dpt.models import DPTDepthModel
except ImportError:
    print('DPT is not installed')
import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from transforms.registry import TRANSFORMS


@lru_cache
def dpt_depth_model(model_path):
    model = DPTDepthModel(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.cuda()
    return model


@ TRANSFORMS.register
class DPTDepth(nn.Module):
    """ ensure multiple of 32, origin net_w = net_h = 384
        set scale to normalize depth and resize to input """

    def __init__(self, pairs, scale=None, gamma=False,
                 model_path='../DPT/weights/dpt_hybrid-midas-501f0c75.pt'):
        super().__init__()
        self.scale_inv = 1 / scale if scale else None
        self.pairs = pairs
        self.gamma = gamma

        self.model = dpt_depth_model(model_path)

    def __call__(self, data):
        for img_name, depth_name in self.pairs:
            img = data['imgs'][img_name]
            if self.gamma:
                img = img**2.2
            img = (img - 0.5) * 2
            d = self.model.forward(img)
            if self.scale_inv:
                d = F.interpolate(
                    d.unsqueeze(1) * self.scale_inv,
                    size=img.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )
            data['imgs'][depth_name] = d

        return data

