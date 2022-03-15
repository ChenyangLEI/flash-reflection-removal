from omegaconf import OmegaConf

DATA_CFG = {
    'data_path':'../data'
}

DATA_TRANSFORMS = {
    'srgb_trans': [
        ["ToTensor"],
    ],
}

DATASETS=OmegaConf.create({
    'synref_train':  {
        "phase": "train",
        "dataset_type": "SrgbLmdbDataset",
        "dataset_args": {
            "path": '${DATA_CFG.data_path}/synthetic/with_syn_reflection/train',
            "ls": 'train.csv'
        },
        "transforms": '${DATA_TRANSFORMS.srgb_trans}'
    },

    'corref_train':  {
        "phase": "train",
        "dataset_type": "SrgbLmdbDataset",
        "dataset_args": {
            "path": '${DATA_CFG.data_path}/synthetic/with_corrn_reflection/train',
            "ls": 'train.csv'
        },
        "transforms": '${DATA_TRANSFORMS.srgb_trans}'
    },

    'real_train':  {
        "phase": "train",
        "dataset_type": "SrgbLmdbDataset",
        "dataset_args": {
            "path": '${DATA_CFG.data_path}/real_world/train',
            "ls": 'train.csv',
            "save_freq": 4,
        },
        "transforms": '${DATA_TRANSFORMS.srgb_trans}'
    },

    'synref_val':  {
        "phase": "val",
        "dataset_type": "SrgbLmdbDataset",
        "dataset_args": {
            "path": '${DATA_CFG.data_path}/synthetic/with_syn_reflection/test',
            "ls": 'test.csv'},
        "transforms": '${DATA_TRANSFORMS.srgb_trans}'
    },
    'corref_val':  {
        "phase": "val",
        "dataset_type": "SrgbLmdbDataset",
        "dataset_args": {
            "path": '${DATA_CFG.data_path}/synthetic/with_corrn_reflection/test',
            "ls": 'test.csv'},
        "transforms": '${DATA_TRANSFORMS.srgb_trans}'
    },

    'real_val':  {
        "phase": "val",
        "dataset_type": "SrgbLmdbDataset",
        "dataset_args": {
            "path": '${DATA_CFG.data_path}/real_world/val',
            "ls": 'val.csv',
            "save_freq": 2,
        },
        "transforms": '${DATA_TRANSFORMS.srgb_trans}'
    },

    'real_test':  {
        "phase": "test",
        "dataset_type": "SrgbLmdbDataset",
        "dataset_args": {
            "path": '${DATA_CFG.data_path}/real_world/test',
            "ls": 'test.csv'
        },
        "transforms": '${DATA_TRANSFORMS.srgb_trans}'
    }
})