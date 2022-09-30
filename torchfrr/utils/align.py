import numpy as np
import torch
from torch import Tensor
from typing import List
from torch.nn.functional import grid_sample
from torchvision.transforms.functional import _get_perspective_coeffs


def rand_sphere(epsilon=1e-5):
    while True:
        x = np.random.uniform(-1, 1, (3,))
        norm = np.linalg.norm(x)
        if norm < 1 and norm > epsilon:
            return x / norm


def axangle2mat(axis, angle):
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    return np.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])


def rand_rotation(max_angle=np.pi / 180):
    angle = np.random.uniform(-max_angle, max_angle)
    axis = rand_sphere()
    return axangle2mat(axis, angle)


def rand_rotran(max_angle=np.pi / 60, max_tran=0.001):
    ret = np.eye(4)
    ret[:3, :3] = rand_rotation(max_angle)
    ret[:3, 3] = np.random.uniform(-max_tran, max_tran, (3,))
    return ret


def projection(n=0.00001, f=2):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, (f + n) / (f - n), -(2 * f * n) / (f - n)],
                     [0, 0, 1, 0]]).astype(np.float32)


def hom2mat(hom):
    a, b, c, d, e, f, g, h = hom
    return np.array([[a, b, c], [d, e, f], [g, h, 1]])


def mat2hom(mat):
    return mat.flatten()[:8]


def get_rand_hom(shift_range, tl, sz):
    """ return forward and backward homography coefficients 
            forward: first coord to second frame """
    # [topleft, topright, botright, botleft]
    corners = tlsz2corners(tl, sz)
    corners_pert = np.array(
        corners) + np.random.randint(-shift_range, shift_range, size=(4, 2))
    fw = _get_perspective_coeffs(corners, corners_pert)
    bw = _get_perspective_coeffs(corners_pert, corners)
    return fw, bw


def shift_hom(hom, w1a, h1a, w1f, h1f):
    H = hom2mat(hom)
    T1 = np.array([[1, 0, w1a], [0, 1, h1a], [0, 0, 1]])
    T2 = np.array([[1, 0, -w1f], [0, 1, -h1f], [0, 0, 1]])
    H2 = T2 @ H @ T1
    hom = H2.flatten()[:8]
    return hom


def flow_warp(img: Tensor, flow: Tensor):
    if flow.dim() < 4:
        flow = flow.unsqueeze(0)
    coord = coord_grid((0, 0), flow.shape[-2:], img.dtype, img.device)
    # print(coord.shape, flow.shape, flush=True)
    grid = flow.permute(0, 2, 3, 1) + coord
    return abs_grid_sample(img, grid)


def abs_grid_sample(img: Tensor, grid, mode='bilinear', padding_mode='zeros'):
    no_batch = False
    if img.dim() < 4:
        img = img.unsqueeze(0)
        no_batch = True
    H, W = img.shape[-2:]
    rel_grid = torch.empty_like(grid)
    rel_grid[..., 0] = grid[..., 0] * (2. / max(W - 1, 1)) - 1.
    rel_grid[..., 1] = grid[..., 1] * (2. / max(H - 1, 1)) - 1.
    img = grid_sample(img, rel_grid, mode, padding_mode, align_corners=False)
    if no_batch:
        img = img.squeeze(0)
    return img


def mesh_grid(x_grid, y_grid):
    w, h = len(x_grid), len(y_grid)
    base_grid = torch.empty(
        1, h, w, 2, dtype=x_grid.dtype, device=x_grid.device)

    base_grid[..., 0].copy_(x_grid)
    base_grid[..., 1].copy_(y_grid.unsqueeze(-1))
    return base_grid


def coord_grid(tl, sz, dtype: torch.dtype, device: torch.device):
    h0, w0 = tl
    oh, ow = sz

    base_grid = torch.empty(1, oh, ow, 2, dtype=dtype, device=device)
    x_grid = torch.arange(w0, w0 + ow, device=device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.arange(h0, h0 + oh, device=device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    return base_grid


def perspective_grid(coeffs: List[float], tl, sz, dtype: torch.dtype, device: torch.device) -> Tensor:
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    h0, w0 = tl
    oh, ow = sz
    theta1 = torch.tensor(
        [[[coeffs[0], coeffs[1], coeffs[2]], [coeffs[3], coeffs[4], coeffs[5]]]
         ], dtype=dtype, device=device
    )
    theta2 = torch.tensor([[[coeffs[6], coeffs[7], 1.0], [
                          coeffs[6], coeffs[7], 1.0]]], dtype=dtype, device=device)

    base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.arange(w0, w0 + ow, device=device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.arange(h0, h0 + oh, device=device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2)
    output_grid1 = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(1, oh * ow, 3).bmm(theta2.transpose(1, 2))

    output_grid = output_grid1 / output_grid2
    return output_grid.view(1, oh, ow, 2)


def tlsz2corners(tl, sz):
    """ get corners from topleft and size
    return [topleft, topright, botright, botleft]
    """
    height, width = sz
    h0, w0 = tl
    return [[w0, h0], [w0 + width - 1, h0], [w0 + width - 1, h0 + height - 1], [w0, h0 + height - 1]]


def rand_offset(shape_origin, shape_crop):
    h, w = shape_origin
    hc, wc = shape_crop

    i = np.random.randint(h - hc + 1)
    j = np.random.randint(w - wc + 1)

    return i, j
