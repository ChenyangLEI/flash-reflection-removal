from omegaconf import OmegaConf
from config.omega.srgb2srgb_fo import _C


_C = _C.copy()

_C.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo']]],
    ['ImgCmds',
        [
            [
                "fo=ab+fo"
            ]
        ]
     ],
    ["ClampImgs", [['ab_T', 'ab_R', 'ab', 'fo']]],
    ['GammaCorrection', [['ab_T', 'ab_R', 'ab', 'fo']]]

])
