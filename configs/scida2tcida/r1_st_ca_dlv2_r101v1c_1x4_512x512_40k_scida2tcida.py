_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/deeplabv3plus_r50-d8.py",
    "../_base_/datasets/st_scida2tcida_512x512.py",
    # Basic UDA Self-Training
    "../_base_/uda/st.py",
    # AdamW Optimizer
    "../_base_/schedules/adamw.py",
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    "../_base_/schedules/poly10warm.py",
    # Schedule
    "../_base_/schedules/schedule_40k.py",
]
# Random Seed
seed = 0
# Fine class path domain bridging with class-mix
data = dict(samples_per_gpu=4, workers_per_gpu=4, train=dict(mask="class"))
# Optimizer Hyper-parameters
optimizer_config = None
optimizer = dict(
    lr=6e-05, paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0)))
)
n_gpus = 1
# Meta Information for Result Analysis
exp = "st"

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', save_best = 'mIoU', by_epoch=False, interval=8000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
