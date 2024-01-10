# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='DeconvModule'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DltMSELoss', loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=128,
    #     in_index=3,
    #     channels=64,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))

crop_size = [384, 384]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    dict(type='LoadMyAnnotations', **img_norm_cfg),
    # dict(type='Resize', img_scale=(768, 768), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    # dict(type='DefaultFormatBundle'),
    dict(type='MyFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

dataset_type = 'LineRemoveDataset'
evaluation = dict(interval=4000, metric='mIoU', pre_eval=False)
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)

lr_config = dict(by_epoch=False, min_lr=0.0001, policy='poly', power=0.9)
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
optimizer_config = dict()
resume_from = None
runner = dict(max_iters=40000, type='IterBasedRunner')

data_root = '/home/andy/Downloads/private/mmsegmentation/demo/data'

data = dict(
    train=dict(
        type = dataset_type,
        data_root = data_root,
        img_dir = 'img_dir',
        ann_dir = 'ann_dir',
        pipeline = train_pipeline,
        split = 'splits/train.txt',
    ),
    samples_per_gpu = 8,
    workers_per_gpu = 1
)
checkpoint_config = dict(by_epoch=False, interval=4000)
workflow = [
    (
        'train',
        1,
    ),
]