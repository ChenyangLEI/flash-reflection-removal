from omegaconf import OmegaConf
from config.omega.lrgb2lrgb_fo import _C

_C = _C.copy()
_C.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo']]],
    ["ClampImgs", [['ab_T', 'ab_R', 'ab', 'fo']]],
    ["GammaCorrection", [['ab_T', 'ab_R', 'ab', 'fo']]]
])
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
                ["perc_rgb_t", 1], ["perc_rgb_r", 1]
            ]
        ]
    ]
])

_C.TRAIN.METRICS = OmegaConf.create([
    ['ClampImgs', [["ab_R_pred", "ab_T_pred"]]],
    [
        'ImgsPsnr',
        [
            [
                ["rgb_t", "ab_T", "ab_T_pred"],
                ["rgb_r", "ab_R", "ab_R_pred"],
            ],
        ],
    ],
    ['EpochImgsWrite', [], {
        'root': '${TRAINDIR}',
        'save_freq': 100,
        'scale': 0.5
    }],
    ['StepMetricsLog']
])
_C.VAL.METRICS = OmegaConf.create([
    ['ClampImgs', [["ab_R_pred", "ab_T_pred"]]],
    [
        'ImgsPsnr',
        [
            [
                ["rgb_t", "ab_T", "ab_T_pred"],
                ["rgb_r", "ab_R", "ab_R_pred"],
            ],
        ],
    ],
    ['EpochImgsWrite', [], {
        'root': '${TRAINDIR}',
        'save_freq': 20,
        'scale': 0.5
    }],
    ['EpochMetricsLog', ['${TRAINDIR}']]
])
_C.TEST.METRICS = OmegaConf.create([
    ['ClampImgs', [["ab_R_pred", "ab_T_pred"]]],
    [
        'ImgsPsnr',
        [
            [
                ["rgb_t0", "ab_T", "ab"],
                ["rgb_t", "ab_T", "ab_T_pred"],
                ["rgb_r", "ab_R", "ab_R_pred"],
            ], 
        ],
    ],
    [
        'ImgsLpips',
        [
            [
                ["rgb_t0", "ab_T", "ab"],
                ["rgb_t", "ab_T", "ab_T_pred"],
                ["rgb_r", "ab_R", "ab_R_pred"],
            ],
        ],
    ],
    [
        'ImgsSsim',
        [
            [
                ["rgb_t0", "ab_T", "ab"],
                ["rgb_t", "ab_T", "ab_T_pred"],
                ["rgb_r", "ab_R", "ab_R_pred"],
            ],
        ],
    ],
    ['EpochImgsWrite', [], {
        'root': '${TRAINDIR}',
        'prefix': '${TEST.NAME}',
        'img_names': ['ab', 'fo', 'ab_R', 'ab_T', 'ab_R_pred', 'ab_T_pred'],
        'save_freq': 1,
        'scale': 1
    }],
    ['EpochMetricsLog', ['${TRAINDIR}', '${TEST.NAME}']]
])

_C.EVAL.METRICS = OmegaConf.create([
    ['ClampImgs', [["ab_R_pred", "ab_T_pred"]]],
    ['EpochImgsWrite', [], {
        'root': '${TRAINDIR}',
        'prefix': '${TEST.NAME}',
        'save_freq': 1,
        'scale': 1
    }],
])
