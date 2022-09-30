from transforms.registry import TRANSFORMS


@TRANSFORMS.register
class GammaCorrection:
    def __init__(self, img_names=None, gamma=1 / 2.2) -> None:
        self.img_names = img_names
        self.gamma = gamma

    def __call__(self, data):
        if self.img_names is None:
            self.img_names = tuple(data['imgs'].keys())
        for img_name in self.img_names:
            data['imgs'][img_name] = data['imgs'][img_name]**self.gamma

        return data


@TRANSFORMS.register
class InverseGamma(GammaCorrection):
    def __init__(self, img_names, gamma=1 / 2.2) -> None:
        super().__init__(img_names, 1 / gamma)


class FilterImgs:
    def __init__(self, filters) -> None:
        self.filters = filters

    def __call__(self, data):
        data['imgs'] = {k: data['imgs'][k] for k in self.filters}

        return data

@TRANSFORMS.register
class DimImgs:
    def __init__(self, pairs) -> None:
        self.pairs = pairs

    def __call__(self, data):
        for name, ratio in self.pairs:
            if data['imgs'][name].max() > ratio:
                data['imgs'][name] *= ratio

        return data


@TRANSFORMS.register
class ImgCmds:
    def __init__(self, cmds) -> None:
        self.cmds = compile('\n'.join(cmds), 'ImgCmds', 'exec')

    def __call__(self, data):
        exec(self.cmds, globals(), data['imgs'])

        return data




@ TRANSFORMS.register
class RenameImgs:
    def __init__(self, pairs) -> None:
        self.pairs = pairs

    def __call__(self, data):
        for src, dst in self.pairs:
            data['imgs'][dst] = data['imgs'].pop(src)

        return data
