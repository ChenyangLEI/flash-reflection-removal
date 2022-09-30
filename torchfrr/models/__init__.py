from torch import nn
from transforms.registry import TRANSFORMS

from models.registry import MODELS
import models.pwcnet


def transform2module(trans):
    class Mod(nn.Module):
        def __init__(self,*args,**kwargs) -> None:
            super().__init__()
            self.trans=trans(*args,**kwargs)
        def forward(self,data):
            return self.trans(data)
    return Mod

def get_model(name, args=(), kwargs={}):
    # print(name, args, flush=True)
    if name in MODELS:
        return MODELS[name](*args, **kwargs)
    else:
        trans=TRANSFORMS[name]
        if not issubclass(trans,nn.Module):
            trans=transform2module(trans)
        return trans(*args, **kwargs)


def get_models(mls):
    if len(mls) == 1:
        return get_model(*mls[0])
    else:
        return nn.Sequential(*[get_model(*m) for m in mls])
