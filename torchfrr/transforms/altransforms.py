import cv2 as cv
import numpy as np
import torch
from utils.align import (abs_grid_sample, coord_grid, flow_warp, get_rand_hom,
                         hom2mat, mat2hom, perspective_grid, rand_offset,
                         shift_hom)

from transforms.registry import TRANSFORMS


@TRANSFORMS.register
class SIFTHom:
    def __init__(self, trip=('ab', 'fo', 'hom_sift_tf'), match_ratio=0.7):
        super().__init__()
        self.sift = cv.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)
        self.match_ratio = match_ratio
        self.trip = trip

    def __call__(self, data):
        if self.trip[0] not in data['imgs']:
            return data
        imgs = (data['imgs'][x] for x in self.trip[:2])
        imgs = [(x / x.max())**(1 / 2.2) * 255 for x in imgs]

        if len(imgs[0].shape) > 3:
            imgs = [x[0] for x in imgs]

        if isinstance(imgs[0], torch.Tensor):
            imgs = (x.permute(1, 2, 0).cpu().numpy() for x in imgs)

        imgs = [np.clip(x, 0, 255).astype(np.uint8)
                for x in imgs]

        kpdes = [self.sift.detectAndCompute(x, None) for x in imgs]
        if (len(kpdes[0][0]) < 4) or (len(kpdes[1][0]) < 4):
            hom = (1, 0, 0, 0, 1, 0, 0, 0)
            data['save_freq'] = torch.tensor([1])
            data['hom'][self.trip[2]] = hom
            data['invalid'] = True
            return data
        matches = self.flann.knnMatch(*[x[1] for x in kpdes[:2]], k=2)
        good = [m for m, n in matches if m.distance <
                self.match_ratio * n.distance]
        if len(good) < 4:
            hom = (1, 0, 0, 0, 1, 0, 0, 0)
            data['save_freq'] = torch.tensor([1])
            data['hom'][self.trip[2]] = hom
            data['invalid'] = True
            return data

        src_pts = np.float32(
            [kpdes[0][0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kpdes[1][0][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if M is None:
            hom = (1, 0, 0, 0, 1, 0, 0, 0)
            data['save_freq'] = torch.tensor([1])
            data['invalid'] = True
        else:
            hom = M.flatten()[:8]
        data['hom'][self.trip[2]] = hom
        return data




@TRANSFORMS.register
class RandHom:

    def __init__(self, shift_range=8, src='ab_T', base_hom='hom_sift_tf', out='rand_sift_tf') -> None:
        """ trip: hom name, src name, flow name """
        self.shift_range = shift_range
        self.src = src
        self.base_hom = base_hom
        self.out = out

    def __call__(self, data):
        base_hom = hom2mat(data['hom'][self.base_hom])
        src = data['imgs'][self.src]
        fw, bw = get_rand_hom(self.shift_range, (0, 0), src.shape[-2:])
        data['hom'][self.out] = mat2hom(hom2mat(fw) @ base_hom)
        data['hom'][self.base_hom] = bw
        return data


@TRANSFORMS.register
class HomFlow:

    def __init__(self, trip=('hom_sift_tf', 'ab', 'flow_tf_sift')) -> None:
        """ trip: hom name, src name, flow name """
        self.trip = trip

    def __call__(self, data):

        hom = np.array(data['hom'][self.trip[0]]).flat
        src = data['imgs'][self.trip[1]]
        tl = (0, 0)
        shape_crop = src.shape[-2:]
        grid = perspective_grid(hom, tl, shape_crop, src.dtype, src.device)
        flowtf = grid - coord_grid(tl, shape_crop, src.dtype, src.device)
        data['imgs'][self.trip[2]] = flowtf.squeeze().permute(2, 0, 1)

        return data


@TRANSFORMS.register
class FlowWarp:

    def __init__(self, dst='fo', flow='flow_tf', out='wfo') -> None:
        """ flow is pixel displacement from src to dst
            warp dst to src and save to out """
        self.dst = dst
        self.flow = flow
        self.out = out

    def __call__(self, data):
        imgs = data['imgs']
        warped = flow_warp(imgs[self.dst], imgs[self.flow])
        data['imgs'][self.out] = warped

        return data



@TRANSFORMS.register
class RandHomRT:
    """ No flash, set fo to marab
        add T and R of ab
        ab = ab_T + ab_R
        srab = ab_T + sab_R
        """

    def __init__(self, shift_range=8, ratio=1, gcd=64) -> None:
        self.gcd = gcd
        self.ratio = ratio
        self.shift_range = shift_range

    def __call__(self, data):

        ab_T, ab_R = (data['imgs'][k] for k in ('ab_T', 'ab_R'))

        shape_T = np.array(ab_T.shape[-2:])
        shape_R = np.array(ab_R.shape[-2:])
        shape_min = np.minimum(shape_T, shape_R)
        padding = self.shift_range
        shape_crop = np.minimum(shape_min - 2 * padding,
                                (shape_min * self.ratio).astype(np.int64))

        if self.gcd != 1:
            shape_crop = shape_crop // self.gcd * self.gcd

        hc, wc = shape_crop

        tls = [np.array(rand_offset(x - 2 * padding, shape_crop)
                        ) + padding for x in (shape_T, shape_R)]
        tlt, tlr = tls

        homs = [get_rand_hom(self.shift_range, x, shape_crop)
                for x in tls]

        gridtb, gridrb = [perspective_grid(
            hom[1], tl, shape_crop, ab_T.dtype, ab_T.device) for hom, tl in zip(homs, tls)]

        gridtf = perspective_grid(
            homs[0][0], tlt, shape_crop, ab_T.dtype, ab_T.device)
        flowtf = gridtf - \
            coord_grid(tlt, shape_crop, ab_T.dtype, ab_T.device)
        data['imgs']['flow_tf'] = flowtf.squeeze().permute(2, 0, 1)

        data['imgs']['sfo'] = abs_grid_sample(data['imgs']['fo'], gridtb)
        data['imgs']['sab_T'] = abs_grid_sample(data['imgs']['ab_T'], gridtb)
        data['imgs']['sab_R'] = abs_grid_sample(data['imgs']['ab_R'], gridrb)

        it, jt = tlt
        ir, jr = tlr
        for tn in ('fo', 'ab_T'):
            data['imgs'][tn] = data['imgs'][tn][..., it:it + hc, jt:jt + wc]
        data['imgs']['ab_R'] = data['imgs']['ab_R'][..., ir:ir + hc, jr:jr + wc]

        return data


@ TRANSFORMS.register
class CropAF:
    """ Crop shift non-shift pair and unify exposure according to metadata
        """

    def __init__(self, ls_name='trip_ma.csv', shift_ls=('sf', 'sab_T'), static_ls=('ab_T', 'ab_R'),
                 hom=['hom_sift_tf', 1], ev_ls=[], gcd=64, padding=8) -> None:
        self.ls_name = ls_name
        self.gcd = gcd
        self.static_ls = static_ls
        self.shift_ls = shift_ls
        self.hom = hom[0]
        self.hom_opt = hom[1]
        self.ev_ls = ev_ls
        self.padding = padding

    def round_crop(self, x, h1, h2, w1, w2):
        h1 = h1 + self.padding
        h2 = h2 + self.padding
        h2 = h1 + (h2 - self.padding - h1) // self.gcd * self.gcd
        w2 = w1 + (w2 - self.padding - w1) // self.gcd * self.gcd
        x = x[..., h1:h2, w1:w2]

        return x

    def __call__(self, data):
        meta = data['meta'][self.ls_name]
        w1a, w2a, h1a, h2a, w1f, w2f, h1f, h2f = (
            meta[x] for x in ('w1a', 'w2a', 'h1a', 'h2a',
                              'w1f', 'w2f', 'h1f', 'h2f'))
        ratio = meta['ev'] / meta['sev']
        if self.hom in data['hom']:
            if self.hom_opt == 0:
                data['hom'][self.hom] = (1, 0, 0, 0, 1, 0, 0, 0)
            elif self.hom_opt == 1:
                data['hom'][self.hom] = shift_hom(
                    data['hom'][self.hom], w1a, h1a, w1f, h1f)
            elif self.hom_opt == 2:
                data['hom'][self.hom] = shift_hom(
                    data['hom'][self.hom], w1a, h1a, w1a, h1a)

        for keya in self.static_ls:
            data['imgs'][keya] = self.round_crop(
                data['imgs'][keya], h1a, h2a, w1a, w2a)
        for keyf in self.shift_ls:
            data['imgs'][keyf] = self.round_crop(
                data['imgs'][keyf], h1f, h2f, w1f, w2f) * ratio
        for keye in self.ev_ls:
            data['imgs'][keye] = data['imgs'][keye] * ratio
        return data

