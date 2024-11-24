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
        'utils.metric'
    ],  
    allow_failed_imports=False 
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
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]
# training schedule for 20k
"""
iter를 기준으로 훈련 진행 
* 주요 계산식 
  - iters_per_epoch = train_size / batch_size
  - max_iters = iters_per_epoch * epochs
  - val_interval = iters_per_epoch * interval
예를 들어, 훈련 데이터 숫자가 640개이고, 훈련 배치 사이즈를 4로 설정하면 iters_per_epoch = 640 / 4 = 160
100 epochs 학습을 원할 경우 16,000 = 160 * 100를 max_iters로 지정
10 epoch마다 검증을 수행할 경우 1,600 = 160 * 10을 val_interval로 지정
"""
train_cfg = dict(type='IterBasedTrainLoop', 
                 max_iters=16000, 
                 val_interval=800)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)