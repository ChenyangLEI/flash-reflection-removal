from omegaconf import OmegaConf
from config.alignment.srgb2srgb_f_rh import _C

_C = _C.copy()


_C.READ_TRANSFORMS = OmegaConf.create([
    ['ToCuda', [['ab_R', 'ab', 'ab_T', 'fo',  'flow_tf']]],
    ['FlowWarp', ['fo', 'flow_tf', 'fo']],
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
_C.TEST.READ_TRANSFORMS = OmegaConf.create([
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
    ['ToCuda', [[ 'fo', 'ab','f']]],
   
    ['SIFTHom', [['ab', 'fo', 'hom_sift_tf']]],
    ['HomFlow', [['hom_sift_tf', 'ab', 'flow_tf_pred']]],
    ['FlowWarp', ['fo', 'flow_tf_pred', 'fo']],
    ['ImgCmds',
     [
         [
             "fo=fo-ab"
         ]
     ]
     ],
    ["ClampImgs", [['fo', ]]],
    ['GammaCorrection', [['f', 'ab', 'fo']]],
    
])
