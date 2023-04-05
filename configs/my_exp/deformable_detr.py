_base_ = '../deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py'

data_root = 'dataset/'

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# dataset settings
dataset_type = 'CocoDataset'
classes = ('car',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'train/annotations/train.json',
        classes=classes,
        img_prefix=data_root + 'train/images/'),
    val=dict(
        ann_file=data_root + 'train/annotations/val.json',
        classes=classes,
        img_prefix=data_root + 'train/images/'),
    test=dict(
        ann_file=data_root + 'train/annotations/val.json',
        classes=classes,
        img_prefix=data_root + 'train/images/'))

runner = dict(type='EpochBasedRunner', max_epochs=10)
evaluation = dict(interval=2, metric='bbox')

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=2)

# pretrained model
load_from = 'pretrained/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'