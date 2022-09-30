from omegaconf import OmegaConf
from config.alignment.srgb2srgb_fo_rh import _C

_C = _C.copy()

_C.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo', ]]],

    ["ClampImgs", [['ab_T', 'ab_R', 'ab', 'fo',  ]]],
    ['GammaCorrection', [['ab_T', 'ab_R']]],

])
_C.EVAL.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab',  'fo', ]]],
    ["ClampImgs", [[ 'ab', 'fo','f']]],
    ['GammaCorrection', [[ 'f']]],
])
_C.MODEL = [
    ['PWCNet', ['~/.cache/mim/pwcnet_ft_4x1_300k_sintel_final_384x768.py',
                '~/.cache/mim/pwcnet_ft_4x1_300k_sintel_final_384x768.pth']],
    ['ImgCmds',
     [
         [
             "fo=fo-ab"
         ]
     ]
     ],
    ["ClampImgs", [['ab', 'fo'], 1e-6]],
    ['GammaCorrection', [[ 'ab', 'fo']]],
    ['MultiUNet'],
]

_C.TRAIN.LOSSES = OmegaConf.create([
    [
        "ImgsPerceptualLoss",
        [
            [
                ["perc_rgb_t", "ab_T", "ab_T_pred"],
                ["perc_rgb_r", "ab_R", "ab_R_pred"],
            ],
        ],
    ],
    [
        "LossesWeightSum",
        [
            [
                ["perc_rgb_t", 1], ["perc_rgb_r", 1], ['flow', 0.03]
            ]
        ]
    ]
])