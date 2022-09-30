import logging

import torch
import torch.nn.functional as F
from mmflow.apis import init_model
from torch import nn

from transforms.registry import TRANSFORMS


@ TRANSFORMS.register
class PWCFlow(nn.Module):
    def __init__(self, cfg, ckpt=None, trip=('ab_T', 'sab_T', 'flow_tf'), gamma=True):
        super().__init__()
        self.trip = trip
        self.cfg = cfg
        self.ckpt = ckpt
        self.encoder = None
        self.decoder = None
        self.gamma = gamma

    def init(self):
        logging.info('lazy init pwcflow')
        model = init_model(self.cfg, self.ckpt, device='cuda')
        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, data):
        imgs = data['imgs']
        if self.trip[2] in imgs or self.trip[0] not in imgs:
            return data
        if self.encoder is None:
            self.init()

        H, W = imgs['ab'].shape[-2:]

        pair = (imgs[x] for x in self.trip[:2])
        if self.gamma:
            pair = (x**(1 / 2.2) for x in pair)

        feat1, feat2 = [self.encoder(x)
                        for x in pair]
        flow_pred = self.decoder.forward(feat1, feat2)
        data['losses']['flow'] = self.decoder.losses(
            flow_pred, imgs['flow_tf'])['loss_flow'] if 'flow_tf' in imgs else torch.tensor(0, dtype=torch.float32, device='cuda')

        flow_result = flow_pred[self.decoder.end_level] * self.decoder.flow_div

        # resize flow to the size of images after augmentation.
        flow_result = F.interpolate(
            flow_result, size=(H, W), mode='bilinear', align_corners=False)

        data['imgs'][self.trip[2]] = flow_result

        return data
