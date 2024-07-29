custom_imports = dict(imports=['mmseg.datasets', 'mmseg.models'], allow_failed_imports=False)

sub_model_train = [
    'panoptic_head',
    'data_preprocessor',
    'backbone'
]

sub_model_optim = {
    'panoptic_head': {'lr_mult': 1},
    'backbone': {'lr_mult': 1}
}

base_lr = 0.0002
# base_lr = 0.001
max_epochs = 200


#### AMP training config
# runner_type = 'Runner'
# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     dtype='float16',
#     optimizer=dict(
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=0.05)
# )

optimizer = dict(
    type='AdamW',
    sub_model=sub_model_optim,
    # lr=0.0005,
    lr=base_lr,
    # weight_decay=1e-3,
    weight_decay=0.05
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.001,
        # eta_min=base_lr,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True
    )
]

# param_scheduler = [
#     # warm up learning rate scheduler
#     dict(
#         type='LinearLR',
#         start_factor=1e-4,
#         by_epoch=True,
#         begin=0,
#         end=1,
#         # update by iter
#         convert_to_iter_based=True),
#     # main learning rate scheduler
#     dict(
#         type='CosineAnnealingLR',
#         T_max=max_epochs,
#         by_epoch=True,
#         begin=1,
#         end=max_epochs,
#     ),
# ]

param_scheduler_callback = dict(
    type='ParamSchedulerHook'
)

evaluator_ = dict(
        type='CocoPLMetric',
        metric=['bbox', 'segm'],
        # metric=['segm'],
        proposal_nums=[1, 10, 100]
)

evaluator = dict(
    val_evaluator=evaluator_,
    test_evaluator=evaluator_,

)


# image_size = (1024, 1024)
image_size = (256, 256)

data_preprocessor1 = dict(
    type='mmdet.DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
)

data_preprocessor2 = dict(
    type='mmdet.DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
)

num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
prompt_shape = (90, 4) #(90, 4)

model_cfg = dict(
    type='SegSAMAnchorPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    need_train_names=sub_model_train,
    data_preprocessor1=data_preprocessor1,
    data_preprocessor2=data_preprocessor2,
    backbone=dict(
        # type='vit_h',
        # checkpoint=r'D:\segment-anything-main\sam_vit_h_4b8939.pth',
        type='vit_b',
        checkpoint=r'D:\segment-anything-main\sam_vit_b_01ec64.pth',
        # checkpoint=r"D:\RSPrompter-cky\tools\results\bijie_rgbd_samus_ins\E20230629_aug\checkpoints\epoch_epoch=197-map_valsegm_map_0=0.3240.ckpt",
    ),
    panoptic_head=dict(
        type='SAMAnchorInstanceHead',
        neck=dict(
            type='SAMAggregatorNeck',
            # in_channels=[1280] * 32,
            in_channels=[768] * 12,
            inner_channels=32,
            # selected_channels=range(4, 32, 2),
            # selected_channels=range(4, 12, 2),
            # selected_channels=range(0, 12, 1),
            selected_channels=[2, 5, 8, 11],
            out_channels=256,
            up_sample_scale=4,
        ),
        # se=dict(
        #     type='SELayer',
        #     channels=512,
        # ),
        # se1=dict(
        #     type='SELayer',
        #     channels=1536,
        # ),
        rpn_head=dict(
            type='mmdet.RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='mmdet.AnchorGenerator',
                scales=[2, 4, 8, 16, 32, 64],
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32]),
            bbox_coder=dict(
                type='mmdet.DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='mmdet.SmoothL1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='SAMAnchorPromptRoIHead',
            bbox_roi_extractor=dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8, 16, 32]),
            bbox_head=dict(
                type='mmdet.Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,  ###########
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='mmdet.DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='mmdet.SmoothL1Loss', loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8, 16, 32]),
            mask_head=dict(
                type='SAMPromptMaskHead',
                per_query_point=prompt_shape[1],
                with_sincos=True,
                class_agnostic=True,
                # loss_mask=dict(
                #     type='mmdet.FocalLoss', loss_weight=10.0))),#
                loss_mask=dict(
                    type='mmdet.CrossEntropyLoss', use_mask=True, loss_weight=10.0))),  #
        # model training and testing settings
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=512,
                    # num=1024,
                    pos_fraction=0.5,
                    # pos_fraction=0.75,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=256,
                    # num=512,
                    pos_fraction=0.25,
                    # pos_fraction=0.50,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=1024,
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                # score_thr=0.5,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                # max_per_img=10,
                max_per_img=100,
                mask_thr_binary=0.5)
        )
    )
)

task_name = 'zhejiang_june_samus_ins'
exp_name = 'E20230629_rgbdem_center'
# task_name = 'bijie_rgbd_samus_ins'
# exp_name = 'E20230629_rgbd'
logger = dict(
    type='WandbLogger',
    project=task_name,
    group='sam-anchor',
    name=exp_name
)

# ###############train#####################
# callbacks = [
#     param_scheduler_callback,
#     dict(
#         type='ModelCheckpoint',
#         dirpath=f'results/{task_name}/{exp_name}/checkpoints',
#         save_last=True,
#         mode='max',
#         monitor='valsegm_map_0',
#         save_top_k=5,
#         filename='epoch_{epoch}-map_{valsegm_map_0:.4f}'
#     ),
#     dict(
#         type='LearningRateMonitor',
#         logging_interval='step'
#     )
# ]
###################predict#########################
callbacks = [
    dict(
        type='DetVisualizationHook',
        draw=True,
        interval=1,
        score_thr=0.05,
        show=False,
        wait_time=1.,
        test_out_dir=f'results_cnn_aug_non1/',
    )
]


vis_backends = [dict(type='mmdet.LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    fig_save_cfg=dict(
        frameon=False,
        figsize=(40, 20),
        # dpi=300,
    ),
    line_width=2,
    alpha=0.8
)

trainer_cfg = dict(
    compiled_model=False,
    accelerator="auto",
    strategy="auto",
    # strategy="ddp",
    # strategy='ddp_find_unused_parameters_true',
    # precision='32',
    # precision='16-mixed',
    devices=1,
    default_root_dir=f'results/{task_name}/{exp_name}',
    # default_root_dir='results/tmp',
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=10,
    check_val_every_n_epoch=3,
    benchmark=True,
    # sync_batchnorm=True,
    # fast_dev_run=True,

    # limit_train_batches=1,
    # limit_val_batches=0,
    # limit_test_batches=None,
    # limit_predict_batches=None,
    # overfit_batches=0.0,

    # val_check_interval=None,
    # num_sanity_val_steps=0,
    # enable_checkpointing=None,
    # enable_progress_bar=None,
    # enable_model_summary=None,
    # accumulate_grad_batches=32,
    # gradient_clip_val=15,
    # gradient_clip_algorithm='norm',
    # deterministic=None,
    # inference_mode: bool=True,
    use_distributed_sampler=True,
    # profiler="simple",
    # detect_anomaly=False,
    # barebones=False,
    # plugins=None,
    # reload_dataloaders_every_n_epochs=0,
)


backend_args = None
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='mmdet.Resize', scale=image_size),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', imdecode_backend='tifffile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=image_size),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
    # meta_keys = ('ori_shape', 'img_shape',
    #              'scale_factor'))
]


train_batch_size_per_gpu = 2
train_num_workers = 2
test_batch_size_per_gpu = 1
test_num_workers = 1
persistent_workers = True


# data_parent = r'D:\RSPrompter-cky\data\bijie_rgbd_aug'
# data_parent = r'D:\RSPrompter-cky\data\zhejiang_rgbd'
data_parent = r'D:\RSPrompter-cky\data\zhejiang_june'
# train_data_prefix = 'train2017/'
# val_data_prefix = 'test2017/'
# test_data_prefix = 'train2017/'
train_data_prefix = 'train2024/'
val_data_prefix = 'test2024/'
test_data_prefix = 'val2024/'
# test_data_prefix = 'test2017_filter/'
dataset_type = 'WHUInsSegDataset'


val_loader = dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        # sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            # ann_file='NWPU_instances_val.json',
            # data_prefix=dict(img_path='positive image set'),
            # ann_file='annotations/SSDD_instances_val.json',
            # data_prefix=dict(img_path='imgs'),
            # ann_file='annotations/bijie_test.json',
            # ann_file='annotations24c/zhejiang_test.json',
            ann_file='annotations/zhejiang_test.json',
            # ann_file='annotations/wenzhou_test.json',
            data_prefix=dict(img_path=val_data_prefix + '/image1'),
            test_mode=True,
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=test_pipeline,
            backend_args=backend_args))

test_loader = dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        # sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            # ann_file='NWPU_instances_val.json',
            # data_prefix=dict(img_path='positive image set'),
            # ann_file='annotations/SSDD_instances_val.json',
            # data_prefix=dict(img_path='imgs'),
            # ann_file='annotations/bijie_train.json',
            # ann_file='annotations24c/zhejiang_val.json',
            ann_file='annotations/zhejiang_val.json',
            # ann_file='annotations_filter/zhejiang_test.json',
            # ann_file='annotations/wenzhou_val.json',
            data_prefix=dict(img_path=test_data_prefix + '/image1'),
            test_mode=True,
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=test_pipeline,
            backend_args=backend_args))


datamodule_cfg = dict(
    type='PLDataModule',
    train_loader=dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        # sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            # ann_file='NWPU_instances_train.json',
            # data_prefix=dict(img_path='positive image set'),
            # ann_file='annotations/SSDD_instances_train.json',
            # data_prefix=dict(img_path='imgs'),
            # ann_file='annotations/bijie_train.json',
            # ann_file='annotations24c/zhejiang_train.json',
            ann_file='annotations/zhejiang_train.json',
            # ann_file='annotations/wenzhou_train.json',
            data_prefix=dict(img_path=train_data_prefix + '/image1'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)
    ),
    val_loader=val_loader,
    test_loader=test_loader,
    predict_loader=test_loader
)