# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Half image resolution

# dataset settings
scida_type = "CidaDataset"
scida_root = "data/ci_da_dataset/"
scida_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
scida_crop_size = (512, 512)

scida_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=scida_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **scida_img_norm_cfg),
    dict(type="Pad", size=scida_crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
train_scida = dict(
    type=scida_type,
    data_root=scida_root,
    img_dir="train_source_image",
    ann_dir="train_source_gt",
    pipeline=scida_train_pipeline,
)
