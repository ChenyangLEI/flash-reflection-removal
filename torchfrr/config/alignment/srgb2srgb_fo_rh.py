from omegaconf import OmegaConf
from config.alignment.srgb2srgb_f_rh import _C

_C = _C.copy()


_C.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo']]],
    ['ImgCmds',
     [
         [
             "fo=fo-ab"
         ]
     ]
     ],
    ["ClampImgs", [['ab_T', 'ab_R', 'ab', 'fo',  ]]],
    ['GammaCorrection', [['ab_T', 'ab_R', 'ab', 'fo']]],
])
_C.EVAL.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab',  'fo', 'f']]],
    ['ImgCmds',
     [
         [
             "fo=fo-ab"
         ]
     ]
     ],
    ["ClampImgs", [[ 'ab', 'fo','f']]],
    ['GammaCorrection', [[ 'f','ab','fo']]],
])
