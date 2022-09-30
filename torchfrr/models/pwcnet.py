import torch.nn as nn
import torch.nn.functional as F
from mmflow.apis import init_model
from utils.align import abs_grid_sample, coord_grid
import torch

from models import MODELS


@MODELS.register
class PWCNet(nn.Module):
    def __init__(self, cfg, ckpt=None):
        super().__init__()
        model = init_model(cfg, ckpt, device='cpu')
        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, data):
        imgs = data['imgs']
        H, W = imgs['ab'].shape[-2:]

        feat1, feat2 = [self.encoder(imgs[x]) for x in ('ab', 'fo')]
        flow_pred = self.decoder.forward(feat1, feat2)
        data['losses']['flow'] = self.decoder.losses(
            flow_pred, imgs['flow_tf'])['loss_flow'] if 'flow_tf' in imgs else torch.tensor(0, dtype=torch.float32, device='cuda')

        flow_result = flow_pred[self.decoder.end_level] * self.decoder.flow_div

        # resize flow to the size of images after augmentation.
        flow_result = F.interpolate(
            flow_result, size=(H, W), mode='bilinear', align_corners=False)

        data['imgs']['flow_tf_pred'] = flow_result
        # reshape [2, H, W] to [H, W, 2]
        flow_result = flow_result.permute(0, 2, 3, 1)
        grid = coord_grid((0, 0), (H, W), flow_result.dtype,
                          flow_result.device) + flow_result

        data['imgs']['fo'] = abs_grid_sample(data['imgs']['fo'], grid)

        return data
