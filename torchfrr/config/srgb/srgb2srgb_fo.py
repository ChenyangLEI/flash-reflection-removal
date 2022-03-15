from omegaconf import OmegaConf
from config.defaults import _C
from config.datasets.srgb import DATA_CFG, DATASETS, DATA_TRANSFORMS

_C = _C.copy()


_C.DATASETS = _C.DATASETS = OmegaConf.masked_copy(DATASETS, [
    "synref_train",
    "corref_train",
    "real_train",
    "synref_val",
    "corref_val",
    "real_val",
    "real_test",
    ]
)
_C.DATA_CFG = DATA_CFG
_C.DATA_TRANSFORMS = DATA_TRANSFORMS
_C.READ_TRANSFORMS = OmegaConf.create([
    ['CropBigImgs',[True,640000]],
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo']]],
])
_C.TEST.READ_TRANSFORMS =  OmegaConf.create([
    ['CropBigImgs',[False,2**22]],
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo']]],
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
_C.VAL.FREQ = 2
_C.VAL.SAVE_FREQ = 20
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
            ], True
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
