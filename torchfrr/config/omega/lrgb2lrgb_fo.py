from omegaconf import OmegaConf
from config.omega.defaults import _C
from config.datasets.lrgb import DATA_CFG, DATASETS, DATA_TRANSFORMS

_C = _C.copy()
_C.DATASETS = OmegaConf.masked_copy(DATASETS, [
    "synref_train",
    "synref_val",
    "corref_train",
    "corref_val",
    "real_train",
    "real_test",
    "real_val"]
)
_C.DATA_CFG = DATA_CFG
_C.DATA_TRANSFORMS = DATA_TRANSFORMS

_C.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo']]],
    ["ClampImgs", [['ab_T', 'ab_R', 'ab', 'fo']]]
])
_C.TRAIN.LOSSES = OmegaConf.create([
    [
        "ImgsPerceptualLoss",
        [
            [
                ["perc_t", "ab_T", "ab_T_pred"],
                ["perc_r", "ab_R", "ab_R_pred"],
            ],
        ],
    ],
    [
        "LossesWeightSum",
        [
            [
                ["perc_t", 1], ["perc_r", 1]
            ]
        ]
    ]
])

_C.TRAIN.METRICS = OmegaConf.create([
    ['ClampImgs', [["ab_R_pred", "ab_T_pred"]]],
    ['GammaCorrection', [['ab', 'fo', 'ab_T', 'ab_R', 'ab_T_pred', 'ab_R_pred']]],
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
        'img_names': ['ab', 'fo', 'ab_T', 'ab_R', 'ab_T_pred', 'ab_R_pred'],
        'save_freq': 100,
        'scale': 0.5
    }],
    ['StepMetricsLog']
])
_C.VAL.FREQ = 2
_C.VAL.SAVE_FREQ = 20
_C.VAL.METRICS = OmegaConf.create([
    ['ClampImgs', [["ab_R_pred", "ab_T_pred"]]],
    ['GammaCorrection', [['ab', 'fo', 'ab_T', 'ab_R', 'ab_T_pred', 'ab_R_pred']]],
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
        'img_names': ['ab', 'fo', 'ab_T', 'ab_R', 'ab_T_pred', 'ab_R_pred'],
        'save_freq': 20,
        'scale': 0.5
    }],
    ['EpochMetricsLog', ['${TRAINDIR}']]
])

_C.TEST.METRICS = OmegaConf.create([
    ['ClampImgs', [["ab_R_pred", "ab_T_pred"]]],
    ['GammaCorrection', [['ab_T', 'ab_R', 'ab', 'fo', "ab_R_pred", "ab_T_pred"]]],
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
