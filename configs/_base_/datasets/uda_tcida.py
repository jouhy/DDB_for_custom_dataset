# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Half image resolution

# dataset settings
tcida_type = "CidaDataset"
tcida_root = "data/ci_da_dataset/"
tcida_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
tcida_crop_size = (512, 512)

tcida_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=tcida_crop_size),  # cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **tcida_img_norm_cfg),
    dict(type="Pad", size=tcida_crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
tcida_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **tcida_img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
train_tcida = dict(
    type=tcida_type,
    data_root=tcida_root,
    img_dir="train_target_image",
    ann_dir="train_target_gt",
    pipeline=tcida_train_pipeline,
)
val_tcida = dict(
    type=tcida_type,
    data_root=tcida_root,
    img_dir="val_s2t_image",
    ann_dir="val_s2t_gt",
    pipeline=tcida_test_pipeline,
)
