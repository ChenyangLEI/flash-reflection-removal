from omegaconf import OmegaConf
from config.alignment.srgb2srgb_f_dw import _C

_C = _C.copy()

_C.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab_R', 'ab_T', 'fo', 'ab']]],
    ["ClampImgs", [['ab_T', 'ab_R', 'ab']]],
    ['DPTDepth', [[
        ['ab_T', 'ab_T_d'],
        ['ab_R', 'ab_R_d'],
    ], 2048, True
    ]],
    ['RandDepthWarp'],
    ['ImgCmds',
     [
         [
             "fo = sfo + sab_T + sab_R"
         ]
     ]
     ],
    ['ImgCmds',
     [
         [
             "fo=fo-ab"
         ]
     ]
     ],
    ["ClampImgs", [['fo', ]]],
    ['GammaCorrection', [['ab_T', 'ab_R', 'ab', 'fo']]],
])

_C.TEST.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo']]],
    ['ImgCmds',
     [
         [
             "fo=fo-ab"
         ]
     ]
     ],
    ["ClampImgs", [['ab_T', 'ab_R', 'ab', 'fo', ]]],
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