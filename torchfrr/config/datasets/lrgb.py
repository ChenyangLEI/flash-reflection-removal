from omegaconf import OmegaConf

DATA_CFG = {
    'aflrgb_train': "data/synthetic/syn_reflection/train/lmdb_lrgb",
    'aflrgb_test': "data/synthetic/syn_reflection/test/lmdb_lrgb",
    'rpath': "data/refreal/lmdb",
    'rtrain': "abR_train.csv",
    'rtest': "abR_test.csv",
    'real_root': 'data/real_world',
    'real2_root': 'data/real2',
    'handheld_root': 'data/handheld'
}

DATA_TRANSFORMS = {
    'synlmdb_trans': [
        ['RandCropThree', [0.8, 320, 32]],
        ['DimImgs', [
            [
                ('ab_T', 0.61),
                ('fo', 0.61),
                ('ab_R', 0.22)
            ]
        ]
        ],
        ['ImgCmds',
         [
             [
                 "ab=ab_R+ab_T"
             ]
         ]
         ],
        ["ToTensor"],
    ],
    'real2_trans': [
        ["ToTensor"],
        ['RandCropImgs', [['ab_R', 'ab_T', 'fo'], 0.8, 320, 32]],
        ['ImgCmds',
         [
             [
                 "ab=ab_R+ab_T"
             ]
         ]
         ],
    ],
}


DATASETS = OmegaConf.create({
    'synref_train': {
        "phase": "train",
        "dataset_type": "SynLmdbDataset",
        "dataset_args": {
            "afpath": '${DATA_CFG.aflrgb_train}',
            "rpath": '${DATA_CFG.aflrgb_train}'
        },
        "transforms": '${DATA_TRANSFORMS.synlmdb_trans}'
    },
    'corref_train': {
        "phase": "train",
        "dataset_type": "SynLmdbDataset",
        "dataset_args": {
            "afpath": '${DATA_CFG.aflrgb_train}',
            "rpath": '${DATA_CFG.rpath}',
            "rls": '${DATA_CFG.rtrain}',
            "dataset_type": "syn_corref"
        },
        "transforms": '${DATA_TRANSFORMS.synlmdb_trans}'
    },
    'real_train': {
        "phase": "train",
        "dataset_type": "Real2Dataset",
        "dataset_args": {
            "root": '${DATA_CFG.real_root}',
            "prefix": 'real_',
            "subdirs": ('train',),
        },
        "transforms": '${DATA_TRANSFORMS.real2_trans}'
    },
    'synref_val': {
        "phase": "val",
        "dataset_type": "SynLmdbDataset",
        "dataset_args": {
            "afpath": '${DATA_CFG.aflrgb_test}',
            "rpath": '${DATA_CFG.aflrgb_test}'},
        "transforms": '${DATA_TRANSFORMS.synlmdb_trans}'
    },
    'corref_val': {
        "phase": "val",
        "dataset_type": "SynLmdbDataset",
        "dataset_args": {
            "afpath": '${DATA_CFG.aflrgb_test}',
            "rpath": '${DATA_CFG.rpath}',
            "rls": '${DATA_CFG.rtest}',
            "dataset_type": "syn_corref"
        },
        "transforms": '${DATA_TRANSFORMS.synlmdb_trans}'
    },
    'real_val': {
        "phase": "val",
        "dataset_type": "Real2Dataset",
        "dataset_args": {
            "root": '${DATA_CFG.real_root}',
            "prefix": 'real_',
            "subdirs": ('val',),
        },
        "transforms": '${DATA_TRANSFORMS.real2_trans}'
    },
    'real_test': {
        "phase": "test",
        "dataset_type": "Real2Dataset",
        "dataset_args": {
            "root": '${DATA_CFG.real_root}',
            "prefix": 'real_',
            "subdirs": ('test',),
            "repeat": 8,
            "save_freq": 1,
        },
        "transforms": '${DATA_TRANSFORMS.real2_trans}'
    },
}
)
