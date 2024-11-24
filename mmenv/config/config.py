# Train Segformer Mit B3
_base_ = [
    "../mmsegmentation/configs/_base_/models/segformer_mit-b0.py",
    "pipeline.py",
    "../mmsegmentation/configs/_base_/default_runtime.py"
]

custom_imports = dict(
    imports=[
        'utils.mixin',
        'utils.dataset',
        'utils.metric',
        'mmseg.models.data_preprocessor'
    ],  
    allow_failed_imports=True
)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0., 0., 0.],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    size=(512, 512),
    pad_val=0,
    seg_pad_val=255,
)

checkpoint="https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth"
model = dict(
    type='EncoderDecoderWithoutArgmax',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 18, 3]),
    decode_head=dict(
        type='SegformerHeadWithoutAccuracy',
        in_channels=[64, 128, 320, 512],
        num_classes=29,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),
)

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# mixed precision
fp16 = dict(loss_scale='dynamic')

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=1e-6, 
        by_epoch=True, 
        begin=0, 
        end=5
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=5,
        end=100,
        by_epoch=True,
    )
]
# training schedule for 20k
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)