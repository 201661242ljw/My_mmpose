# _base_ = [
#     '../../../../_base_/default_runtime.py',
#     '../../../../_base_/datasets/coco.py'
# ]
_base_ = [
    '_base_/default_runtime.py',
    '_base_/datasets/tower_12456.py'
]
checkpoint_config = dict(interval=50)
evaluation = dict(interval=50, metric='mAP', save_best='AP')

batch_size = 2
dataset_part = 0
optimizer = dict(
    type='Adam',
    lr=0.0015,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[200, 260])
total_epochs = 100
num_points = 224
channel_cfg = dict( num_output_channels =num_points,
                    dataset_joints      =num_points,
                    dataset_channel     =[list(range(num_points))],
                    inference_channel   =list(range(num_points))
                    )
log_config = dict(
    interval= 799 // batch_size // 50 * 50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
data_cfg = dict(
    image_size=640,
    base_size=320,
    base_sigma=2,
    heatmap_size=[160],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=1,
    scale_aware_sigma=False,
)

# model settings
model = dict(
    type='AssociativeEmbedding',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='AESimpleHead',
        in_channels=2048,
        num_joints=224,
        tag_per_joint=True,
        with_ae_loss=[True],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=224,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0])),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        scale_factor=[1],
        with_heatmaps=[True],
        with_ae=[True],
        project2image=True,
        align_corners=False,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=True))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=30,
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]

test_pipeline = val_pipeline
# data_root = 'data/coco'
# data_root = '../00_LJW/tower_dataset_12456'
data_root = '../00_LJW/tower_dataset_12456_train_val_test'
data = dict(
    # samples_per_gpu=32,
    # workers_per_gpu=2,
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        # type='TopDownCocoDataset',
        type='TopDownCocoLikeTower12456Dataset',
        ann_file=f'{data_root}/annotations/{dataset_part}_keypoints_train.json',
        img_prefix=f'{data_root}/imgs/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        # type='TopDownCocoDataset',
        type='TopDownCocoLikeTower12456Dataset',
        ann_file=f'{data_root}/annotations/{dataset_part}_keypoints_val.json',
        img_prefix=f'{data_root}/imgs/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        # type='TopDownCocoDataset',
        type='TopDownCocoLikeTower12456Dataset',
        ann_file=f'{data_root}/annotations/{dataset_part}_keypoints_test.json',
        img_prefix=f'{data_root}/imgs/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)