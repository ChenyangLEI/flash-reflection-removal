import os.path as osp
import os

def be(ls,ext):
    return [name[:-4]+'.'+ext for name in ls]

def bj(root,ls):
    return [osp.join(root,f) for f in ls]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
