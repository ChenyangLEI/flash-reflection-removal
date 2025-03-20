from omegaconf import OmegaConf
from config.omega.srgb2srgb_f import _C
from config.datasets.randhomrt import DATA_TRANSFORMS, DATASETS

_C = _C.copy()
_C.DATASETS = OmegaConf.merge(OmegaConf.masked_copy(_C.DATASETS, [
    "synref_train",
    "synref_val",
    "corref_train",
    "corref_val",
    "real_train",
    "real_val",
]) , OmegaConf.masked_copy(DATASETS, [
    "real2ma_test",
    'real2masa_test',
    "handheld_eval",
]))
_C.DATA_TRANSFORMS = OmegaConf.merge(_C.DATA_TRANSFORMS, 
    OmegaConf.masked_copy(OmegaConf.create(DATA_TRANSFORMS), [
        'real2masa_trans',
        'real2ma_trans',
        'handheld_trans',
]))
                                         

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
    ["ClampImgs", [['fo', ]]],
    ['GammaCorrection', [['ab_T', 'ab_R', 'ab', 'fo']]],
])

_C.TEST.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo']]],
    ["ClampImgs", [['ab_T', 'ab_R', 'ab', 'fo' ]]],
    ['GammaCorrection', [['ab_T', 'ab_R', 'ab', 'fo']]],
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
        'img_names': ['ab', 'fo', 'ab_T', 'ab_R', 'ab_T_pred', 'ab_R_pred',
                      'flow_tf', 'flow_tf_pred', 'ab_T_d', 'ab_R_d'],
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
        'img_names': ['ab', 'fo', 'ab_T', 'ab_R', 'ab_T_pred', 'ab_R_pred',
                      'flow_tf', 'flow_tf_pred', 'ab_T_d', 'ab_R_d'],
        'save_freq': 20,
        'scale': 0.5
    }],
    ['EpochMetricsLog', ['${TRAINDIR}']]
])
