data_root = '/data/guxiang/gx/dataset/CarlaData/Town10HD'
ann_file = data_root
camera_used = ['bev']

dataset_type = 'Town10HDDataset'
# Most Town10HD scenes use the evaluation-space bounds below. scene_00 is
# narrower in x, and is remapped into this common training space in the
# dataset loader.
point_cloud_range = [-25.6, -25.6, -32.0, 51.2, 25.6, 8.0]
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [1.0, 81.0, 1.0],
}

class_names = [
    'empty',
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic_light',
    'traffic_sign',
    'vegetation',
    'terrain',
    'sky',
    'pedestrian',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'static',
    'dynamic',
    'other',
    'water',
    'road_line',
    'ground',
    'bridge',
    'rail_track',
    'guard_rail',
]
num_class = len(class_names)

bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0,
    flip_dz_ratio=0.0,
)

data_config = {
    'input_size': (384, 704),
    'resize': (0.0, 0.0),
    'rot': (0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.0,
}

train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        data_config=data_config,
        load_stereo_depth=False,
        is_train=True,
        color_jitter=(0.2, 0.2, 0.2),
    ),
    dict(type='CreateDepthFromLiDARCarla', data_root=data_root, load_seg=True),
    dict(
        type='LoadAnnotationOcc',
        bda_aug_conf=bda_aug_conf,
        apply_bda=False,
        is_train=True,
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type='CollectData',
        keys=['img_inputs', 'gt_semantics'],
        meta_keys=[
            'pc_range',
            'occ_size',
            'raw_img',
            'stereo_depth',
            'gt_depths',
            'sequence',
            'frame_id',
            'img_shape',
            'lidar_path',
            'lidar_label_path',
        ],
    ),
]

test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        data_config=data_config,
        load_stereo_depth=False,
        is_train=False,
        color_jitter=None,
    ),
    dict(type='CreateDepthFromLiDARCarla', data_root=data_root, load_seg=True),
    dict(
        type='LoadAnnotationOcc',
        bda_aug_conf=bda_aug_conf,
        apply_bda=False,
        is_train=False,
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type='CollectData',
        keys=['img_inputs', 'gt_semantics'],
        meta_keys=[
            'pc_range',
            'occ_size',
            'raw_img',
            'stereo_depth',
            'gt_depths',
            'sequence',
            'frame_id',
            'img_shape',
            'lidar_path',
            'lidar_label_path',
        ],
    ),
]

trainset_config = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=train_pipeline,
    split='train',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=False,
    label_merge_mode='max',
    camera_height=60.0,
    num_classes=num_class,
)

testset_config = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    split='val',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=True,
    label_merge_mode='max',
    camera_height=60.0,
    num_classes=num_class,
)

data = dict(
    train=trainset_config,
    val=testset_config,
    test=testset_config,
)

train_dataloader_config = dict(batch_size=4, num_workers=4)
test_dataloader_config = dict(batch_size=4, num_workers=4)

numC_Trans = 128

model = dict(
    type='CGFormerSegDepth',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint='./ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth',
        ),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.5, 1, 2, 4, 4],
        out_channels=[128, 128, 128, 128, 128],
    ),
    depth_net=dict(
        type='GeometryDepth_NetCarla',
        downsample=8,
        numC_input=640,
        numC_Trans=numC_Trans,
        cam_channels=33,
        grid_config=grid_config,
        loss_depth_type='kld',
        loss_depth_weight=0.0001,
    ),
    plugin_head=dict(
        type='plugin_segmentation_head',
        in_channels=numC_Trans,
        out_channel_list=[128, 64, 32],
        num_class=num_class,
    ),
)

learning_rate = 3e-4
training_steps = 50000

optimizer = dict(
    type='AdamW',
    lr=learning_rate,
    weight_decay=0.01,
)

lr_scheduler = dict(
    type='OneCycleLR',
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy='cos',
    interval='step',
    frequency=1,
)
