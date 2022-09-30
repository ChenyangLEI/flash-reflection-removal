from omegaconf import OmegaConf
from config.omega.srgb2srgb_f import _C
from config.datasets.randhomrt import DATA_CFG, DATA_TRANSFORMS, DATASETS

_C = _C.copy()
_C.DATASETS = OmegaConf.masked_copy(DATASETS, [
    "synref_train",
    "synref_val",
    "corref_train",
    "corref_val",
    "real_train",
    "real_val",
    'handheld_eval',
    "real2ma_test",
    'real2masa_test',
]
)

_C.DATA_CFG = DATA_CFG
_C.DATA_TRANSFORMS = DATA_TRANSFORMS
# fo is f here, no need to add
_C.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo']]],
    ["ClampImgs", [['ab_T', 'ab_R', 'ab', 'fo']]],
    ['GammaCorrection', [['ab_T', 'ab_R', 'ab', 'fo']]],
])