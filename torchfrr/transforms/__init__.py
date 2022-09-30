import transforms.altransforms
import transforms.dpttransform
import transforms.drtransforms
import transforms.iotransforms
import transforms.losstransforms
import transforms.mmtransforms
import transforms.nntransforms
import transforms.nptransforms
import transforms.perctransforms
import transforms.pttransforms
import transforms.transforms
from transforms.registry import TRANSFORMS


def get_transform(name, args=(), kwargs={}):
    return TRANSFORMS[name](*args, **kwargs)


def get_transforms(tls):
    return [get_transform(*t) for t in tls]
