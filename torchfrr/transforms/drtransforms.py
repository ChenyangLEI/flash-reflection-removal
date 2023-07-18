try:
    import nvdiffrast.torch as dr
except ImportError:
    print('nvdiffrast not installed')
from functools import lru_cache

import numpy as np
import torch
from utils.align import axangle2mat, mesh_grid, projection, rand_sphere

from transforms.registry import TRANSFORMS


@lru_cache
def get_glctx():
    return dr.RasterizeGLContext()


@TRANSFORMS.register
class RandDepthWarp:
    def __init__(self, max_tran=0.01, max_angle=np.pi / 180,
                 rshift_range=(-0.1, 32), gcd=64):
        self.glctx = get_glctx()
        self.max_tran = max_tran
        self.max_angle = max_angle
        self.rshift_range = rshift_range
        self.gcd = gcd

    @staticmethod
    def disparity2depth(disparity, dshift_range=0):
        """ disparity have maximum around 1,
            which correspond to approximately 0.5 m in real world
            adjust disparity so that depth is in meter between 0.5 to 200 """
        depth = 1 / (disparity * 2 + 0.005)
        # imtshow(depth)
        depth[depth > 100] = 100
        # imtshow(depth)
        depth = depth[..., None]
        # print(depth.shape)
        if dshift_range:
            depth += np.random.uniform(*dshift_range)
        return depth

    @staticmethod
    def grid_triangles(w, h):
        y, x = torch.meshgrid(torch.arange(h - 1, dtype=torch.int32, device='cuda'),
                              torch.arange(w - 1, dtype=torch.int32, device='cuda'))
        tl = y * w + x
        tr = y * w + x + 1
        bl = (y + 1) * w + x
        br = (y + 1) * w + x + 1
        return torch.stack([tl, bl, tr, br, tr, bl], dim=-1).reshape(-1, 3)

    @staticmethod
    def backproject(xy, depth):
        *size, _ = xy.shape
        ret = torch.empty((*size, 4), dtype=xy.dtype, device=xy.device)
        ret[:, :, :, 0:2] = xy * depth
        ret[..., 2:3] = depth
        ret[..., 3] = 1
        return ret

    def rasterize(self, points, faces, h, w):
        with torch.no_grad():
            rast_out, _ = dr.rasterize(self.glctx, points.view(1, -1, 4),
                                       faces, resolution=[h, w], grad_db=False)
        return rast_out

    @staticmethod
    def warp(color, rast, faces):
        color = color.reshape(3, -1).T.contiguous()
        with torch.no_grad():
            cout, _ = dr.interpolate(color, rast, faces)
        return cout

    def __call__(self, data):

        imgs = data['imgs']
        fo = imgs['fo']
        ab_T = imgs['ab_T']
        ab_R = imgs['ab_R']
        h, w = ab_T.shape[-2:]
        depth_T = self.disparity2depth(imgs['ab_T_d'])
        depth_R = self.disparity2depth(imgs['ab_R_d'], self.rshift_range)
        angle = np.random.uniform(0, self.max_angle)
        axis = rand_sphere()
        rt = np.eye(4)
        rt[:3, :3] = axangle2mat(axis, angle)
        rt[:3, 3] = np.random.uniform(-self.max_tran, self.max_tran, (3,))
        data['trans3d']['rtdw'] = rt
        data['trans3d']['angle'] = angle
        data['trans3d']['axis'] = axis
        tran_mat = torch.from_numpy(
            (rt.T @ projection(n=0.1, f=200).T).astype(np.float32)).cuda()
        triangles = self.grid_triangles(w, h)

        # ratio from [-1,1] to w,h
        wh = torch.tensor([w, h], dtype=torch.int32, device='cuda') / 2
        # clip space xy
        xy = mesh_grid(torch.linspace(-1, 1, w, dtype=torch.float32, device='cuda'),
                       torch.linspace(-1, 1, h, dtype=torch.float32, device='cuda'))

        coord_T = self.backproject(xy, depth_T)
        scoord_T = coord_T.reshape(-1, 4) @ tran_mat
        flow_T = (scoord_T[:, :2] / (scoord_T[:, 3:4])
                  ).reshape(1, h, w, 2) - xy
        max_flow_T = flow_T.view(-1, 2).max(dim=0)[0]
        min_flow_T = flow_T.view(-1, 2).min(dim=0)[0]
        flow_T *= wh
        max_flow = flow_T.abs().max()
        data['metrics']['max_flow'] = max_flow

        coord_R = self.backproject(xy, depth_R)
        scoord_R = coord_R.reshape(-1, 4) @ tran_mat
        flow_R = (scoord_R[:, :2] / (scoord_R[:, 3:4])
                  ).reshape(1, h, w, 2) - xy
        max_flow_R = flow_R.view(-1, 2).max(dim=0)[0]
        min_flow_R = flow_R.view(-1, 2).min(dim=0)[0]

        max_flow = torch.maximum(max_flow_T, max_flow_R)
        min_flow = torch.minimum(min_flow_T, min_flow_R)
        bbmin = (torch.clamp(max_flow, min=0) * wh).int()
        bbmax_o = (torch.clamp(2 + min_flow, max=2) * wh).int()
        bbmax = bbmin + (bbmax_o - bbmin) // self.gcd * self.gcd
        assert (bbmax_o >= bbmax).all()
        data['trans3d']['bb'] = (bbmin, bbmax)

        rast_T = self.rasterize(scoord_T, triangles, h, w)
        sab_T = self.warp(ab_T, rast_T, triangles)
        sfo = self.warp(fo, rast_T, triangles)

        rast_R = self.rasterize(scoord_R, triangles, h, w)
        sab_R = self.warp(ab_R, rast_R, triangles)

        for k in ('ab_T', 'ab_R', 'fo', 'ab'):
            data['imgs'][k] = data['imgs'][k][:, :,
                                              bbmin[1]:bbmax[1], bbmin[0]:bbmax[0]]
        data['imgs']['sab_T'] = sab_T.permute(
            0, 3, 1, 2)[:, :, bbmin[1]:bbmax[1], bbmin[0]:bbmax[0]]
        data['imgs']['sab_R'] = sab_R.permute(
            0, 3, 1, 2)[:, :, bbmin[1]:bbmax[1], bbmin[0]:bbmax[0]]
        data['imgs']['sfo'] = sfo.permute(
            0, 3, 1, 2)[:, :, bbmin[1]:bbmax[1], bbmin[0]:bbmax[0]]
        data['imgs']['flow_tf'] = flow_T.permute(
            0, 3, 1, 2)[:, :, bbmin[1]:bbmax[1], bbmin[0]:bbmax[0]]
        return data
