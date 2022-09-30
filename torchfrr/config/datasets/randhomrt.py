from config.datasets.lrgb import DATA_CFG, DATASETS, DATA_TRANSFORMS

DATA_CFG = DATA_CFG.copy()
DATASETS = DATASETS.copy()
DATA_TRANSFORMS = DATA_TRANSFORMS.copy()
"""
    fo here are actually f,
    just use the name fo for convenience
"""
DATASETS.update({
    'handheld_eval': {
        "phase": "eval",
        "dataset_type": "Real2Dataset",
        "dataset_args": {
                "root": '${DATA_CFG.handheld_root}',
                "subdirs": ('wild',),
                "ls_name": 'trip.csv',
                "img_names": ['ab', 'f'],
                "repeat": 1,
                "save_freq": 1,
                },
        "transforms": '${DATA_TRANSFORMS.handheld_trans}'
    },
    'real2ma_test': {
        "phase": "test",
        "dataset_type": "Real2Dataset",
        "dataset_args": {
            "root": '${DATA_CFG.real2_root}',
            "subdirs": ( 'deskalign_0206','deskalign_0218'),
            "ls_name": 'trip_ma.csv',
            "img_names": ['ab_R', 'ab_T', 'sf', 'sab_T'],
            'save_freq': 1,
            "repeat": 1,
            "prefix": "ma"
        },
        "transforms": '${DATA_TRANSFORMS.real2ma_trans}'
    },
    'real2masa_test': {
        "phase": "test",
        "dataset_type": "Real2Dataset",
        "dataset_args": {
            "root": '${DATA_CFG.real2_root}',
            "subdirs": ( 'deskalign_0206','deskalign_0218'),
            "ls_name": 'trip_ma.csv',
            "img_names": ['ab_R', 'ab_T', 'sf', 'sab_T'],
            'save_freq': 1,
            "repeat": 1,
            "prefix": "masa"
        },
        "transforms": '${DATA_TRANSFORMS.real2masa_trans}'
    },
})

DATA_TRANSFORMS.update({
    'synlmdb_trans': [
        ["ToTensor"],
        ['RandHomRT', [8, 0.8, 64]],
        ['DimImgs', [
            [
                ('ab_T', 0.61),
                ('ab_R', 0.22),
                ('sfo', 0.61),
                ('sab_T', 0.61),
                ('sab_R', 0.22),
            ]
        ]
        ],
        ['ImgCmds',
         [
             [
                 "ab = ab_R + ab_T",
                 "fo = sfo + sab_T + sab_R"
             ]
         ]
         ],
    ],

    'real2_trans': [
        ["ToTensor"],
        ['RandHomRT', [8, 0.8, 64]],
        ['ImgCmds',
         [
             [
                 "ab = ab_R + ab_T",
                 "fo = sfo + sab_T + sab_R"
             ]
         ]
         ],
    ],
    'real2ma_trans': [
        ['SIFTHom', [['ab_T', 'sab_T', 'hom_sift_tf']]],
        ["ToTensor"],
        ['CropAF', ['trip_ma.csv', ['sab_T', 'sf'],
                    ('ab_T', 'ab_R', ), ['hom_sift_tf', 1], []]],
        ['ImgCmds',
         [
             [
                 "ab=ab_R+ab_T",
                 "fo=sf",
             ]
         ]
         ],
    ],
    'real2masa_trans': [
        ['SIFTHom', [['ab_T', 'sab_T', 'hom_sift_tf']]],
        ["ToTensor"],
        ['HomFlow', [['hom_sift_tf', 'ab_T', 'flow_tf_sift']]],
        ['FlowWarp', ['sf', 'flow_tf_sift', 'sf']],
        ['FlowWarp', ['sab_T', 'flow_tf_sift', 'sab_T']],

        ['CropAF', ['trip_ma.csv', [],
                    ('ab_T', 'ab_R', 'sab_T', 'sf'), ['hom_sift_tf', 0], ['sf', 'sab_T']]],
        ['ImgCmds',
         [
             [
                 "ab=ab_R+ab_T",
                 "fo=sf",
             ]
         ]
         ],
    ],
    'handheld_trans': [
        ['SIFTHom', [['ab', 'f', 'hom_sift_tf']]],
        ["ToTensor"],
        ["RandCropImgs", [['ab', 'f'], 1, 320, 64]],
        ['HomFlow', [['hom_sift_tf', 'ab', 'flow_tf_sift']]],
        ['ImgCmds',
         [
             [
                 "fo=f",
             ]
         ]
         ],
    ],
})
