batch_size = 2
workers = 2
total_epochs = 100
img_size = 1024
dataset_part = 0
num_points = 224
sigma = 2

_base_ = [
    '_base_/default_runtime.py',
    '_base_/datasets/tower_12456.py'
]
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])

log_config = dict(
    interval=802 // batch_size // 50 * 50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(num_output_channels=num_points,
                   dataset_joints=num_points,
                   dataset_channel=[list(range(num_points))],
                   inference_channel=list(range(num_points))
                   )

data_cfg = dict(
    # image_size=[288, 384],
    # heatmap_size=[72, 96],
    image_size=[img_size, img_size],
    heatmap_size=[img_size // 4, img_size // 4],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    # use_gt_bbox=False,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    # bbox_file='data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
    bbox_file='',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='TopDownGetBboxCenterScale', padding=1.25),
    # dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    # dict(type='TopDownRandomFlip', flip_prob=0.5),
    # dict(type='TopDownHalfBodyTransform', num_joints_half_body=8, prob_half_body=0.3),
    # dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownGetBboxCenterScale', padding=1.1),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.1, prob=0.5),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # dict(type='TopDownGenerateTarget', sigma=2),
    dict(type='TopDownGenerateTarget', sigma=sigma),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.1),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = f'/kaggle/input/tower-dataset-2/resized_dataset/{img_size}'
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=workers,
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

# model settings
pretrained = ('https://github.com/SwinTransformer/storage/releases/download'
              '/v1.0.0/swin_base_patch4_window7_224_22k.pth')

model = dict(
    type='TopDown',
    pretrained=pretrained,
    backbone=dict(
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=1024,
        out_channels=channel_cfg['num_output_channels'],
        in_index=3,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))
