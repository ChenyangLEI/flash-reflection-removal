
from models.registry import MODELS
from torch import nn
from transforms import TRANSFORMS


def get_model(name, args=(), kwargs={}):
    # print(name, args, flush=True)
    if name in MODELS:
        return MODELS[name](*args, **kwargs)
    else:
        return TRANSFORMS[name](*args, **kwargs)


def get_models(mls):
    if len(mls) == 1:
        return get_model(*mls[0])
    else:
        return nn.Sequential(*[get_model(*m) for m in mls])
