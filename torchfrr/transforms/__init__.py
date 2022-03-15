from transforms.registry import TRANSFORMS
import transforms.nntransforms
import transforms.pttransforms
import transforms.iotransforms
import transforms.transforms

def get_transform(name, args=(), kwargs={}):
    # print(name, args, flush=True)
    return TRANSFORMS[name](*args, **kwargs)


def get_transforms(tls):
    return [get_transform(*t) for t in tls]
