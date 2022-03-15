import numpy as np


def rand_offset(shape_origin, shape_crop):
    h, w = shape_origin
    hc, wc = shape_crop

    i = np.random.randint(h - hc + 1)
    j = np.random.randint(w - wc + 1)

    return i, j


