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
    ["ClampImgs", [['fo',  ]]],
    ['GammaCorrection', [['ab_T', 'ab_R']]],
])

_C.TEST.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo']]],
    ["ClampImgs", [['ab_T', 'ab_R', 'ab', 'fo' ]]],
    ['GammaCorrection', [['ab_T', 'ab_R']]],
])
_C.EVAL.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab',  'fo', ]]],
    ["ClampImgs", [[ 'ab', 'fo']]],
    ['ImgCmds',
     [
         [
             "f=fo",
             'mfo=f-ab'
         ]
     ]
     ],
    ['GammaCorrection', [[ 'f','mfo']]],
])

_C.MODEL = [
    ['PWCNet', ['~/.cache/mim/pwcnet_ft_4x1_300k_sintel_final_384x768.py',
                '~/.cache/mim/pwcnet_ft_4x1_300k_sintel_final_384x768.pth']],
    ['ImgCmds',
     [
         [
             'fa=fo',
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
_C.EVAL.METRICS = OmegaConf.create([

    ['GammaCorrection', [[  'fa']]],

    ['EpochImgsWrite', [], {
        'root': '${TRAINDIR}',
        'prefix': '${TEST.NAME}',
        'save_freq': 1,
        'scale': 1
    }],
])
