
import blosc
import numpy as np
import torch


def rgb2qgray(img: torch.Tensor):
    """ Input float tensor with shape b,3,h,w and range [0,1] 
        Return uint8 tensor with shape b,1,h,w and range [0,255] """
    b, _, h, w = img.shape
    img = img.reshape((3, h * w))
    coefficient = torch.tensor(
        [[0.2989, 0.5870, 0.1140]], dtype=img.dtype, device=img.device)
    img = coefficient @ img
    img = img.reshape((b, 1, h, w))
    img = (img * 255
           ).type(torch.uint8)
    return img


def structural_similarity(im1, im2, K1=0.01, K2=0.03, win_size=7, R=255):
    im1, im2 = (rgb2qgray(x).type(torch.float32) for x in (im1, im2))
    numel = win_size**2
    kernel = torch.ones((1, 1, win_size, win_size), dtype=im1.dtype,
                        device=im1.device) / numel
    cov_norm = numel / (numel - 1)

    ux, uy = (torch.conv2d(x, kernel) for x in (im1, im2))
    uxx, uyy = (torch.conv2d(x**2, kernel) for x in (im1, im2))
    uxy = torch.conv2d(im1 * im2, kernel)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D
    ssim = S.mean()
    return ssim


def peak_signal_noise_ratio(image_true, image_test, dmin=0, dmax=1):
    """ Assert image in [0,1] """
    assert image_test.shape == image_true.shape
    assert image_test.dtype == image_true.dtype

    true_min, true_max = image_true.min(), image_true.max()
    if true_max > dmax or true_min < dmin:
        raise ValueError(
            "im_true has intensity values outside the range expected for "
            "its data type.  Please manually specify the data_range")
    if true_min >= 0:
        data_range = dmax
    else:
        data_range = dmax - dmin

    err = (image_true - image_test)**2
    if isinstance(image_true, np.ndarray):
        psnr = 10 * np.log10((data_range ** 2) / err.mean())
    else:
        psnr = 10 * torch.log10((data_range ** 2) / err.mean())
    return psnr, err




def img2float(img):
    """normalize img and convert to float"""
    return (img / np.iinfo(img.dtype).max).astype(np.float32)


def imgs2float(data):
    """normalize imgs and convert to float"""
    for k, v in data['imgs'].items():
        if np.issubdtype(v.dtype, np.integer):
            data['imgs'][k] = (v / np.iinfo(v.dtype).max).astype(np.float32)
    return data
