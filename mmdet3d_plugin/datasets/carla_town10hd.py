import glob
import json
import os

import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset


@DATASETS.register_module()
class Town10HDDataset(Dataset):
    def __init__(
        self,
        data_root,
        ann_file=None,
        pipeline=None,
        split='train',
        camera_used=None,
        occ_size=None,
        pc_range=None,
        stereo_depth_root=None,
        test_mode=False,
        load_continuous=False,
        image_dir='bev',
        lidar_dir='velodyne',
        point_label_dir='labels',
        occ_dir='evaluation',
        occ_label_suffix='.label',
        label_merge_mode='max',
        camera_height=60.0,
        num_classes=None,
        ignore_label=255,
        preprocess_label_subdir='labels',
        preprocess_scale='1_1',
        prefer_preprocessed_labels=False,
        scene_names=None,
    ):
        super().__init__()

        self.data_root = data_root
        self.ann_file = ann_file if ann_file is not None else data_root
        self.test_mode = test_mode
        self.load_continuous = load_continuous
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.stereo_depth_root = stereo_depth_root

        self.image_dir = image_dir
        self.lidar_dir = lidar_dir
        self.point_label_dir = point_label_dir
        self.occ_dir = occ_dir
        self.occ_label_suffix = occ_label_suffix
        self.label_merge_mode = label_merge_mode
        self.camera_height = camera_height
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.preprocess_label_subdir = preprocess_label_subdir
        self.preprocess_scale = preprocess_scale
        self.prefer_preprocessed_labels = prefer_preprocessed_labels
        self.scene_names = None if scene_names is None else set(scene_names)
        self.target_occ_size = np.array(self.occ_size, dtype=np.int32)
        self.target_pc_range = np.array(self.pc_range, dtype=np.float32)

        self.camera_map = {'bev': '0'}
        camera_used = ['bev'] if camera_used is None else camera_used
        self.camera_used = [self.camera_map[camera] for camera in camera_used]

        split_map = {
            'train': 'Train',
            'val': 'Val',
            'test': 'Val',
        }
        if split not in split_map:
            raise ValueError(f'Unsupported split: {split}')
        self.scene_root = os.path.join(self.data_root, split_map[split])

        self.data_infos = self.load_annotations()
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def __len__(self):
        return len(self.data_infos)

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        return self.pipeline(input_dict)

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in test data')
            return None
        return self.pipeline(input_dict)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)

        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def get_data_info(self, index):
        info = self.data_infos[index]

        input_dict = dict(
            occ_size=np.array(self.occ_size),
            pc_range=np.array(self.pc_range),
            sequence=info['sequence'],
            frame_id=info['frame_id'],
            lidar_path=info['lidar_path'],
            lidar_label_path=info['lidar_label_path'],
        )

        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []
        for cam_type in self.camera_used:
            image_paths.append(info[f'img_{int(cam_type)}_path'])
            lidar2img_rts.append(info[f'proj_matrix_{int(cam_type)}'])
            cam_intrinsics.append(info[f'P{int(cam_type)}'])
            lidar2cam_rts.append(info['T_velo_2_cam'])

        focal_length = info['P0'][0, 0]
        baseline = 0.0

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                focal_length=focal_length,
                baseline=baseline,
                stereo_depth_path=None,
            )
        )
        input_dict['gt_occ'] = self.get_ann_info(index, key='voxel_path')
        return input_dict

    def load_annotations(self):
        scans = []
        scene_dirs = sorted(glob.glob(os.path.join(self.scene_root, 'scene_*')))
        for scene_dir in scene_dirs:
            sequence = os.path.basename(scene_dir)
            if self.scene_names is not None and sequence not in self.scene_names:
                continue
            cartesian_dir = os.path.join(scene_dir, 'cartesian')
            image_base_path = os.path.join(cartesian_dir, self.image_dir)
            lidar_base_path = os.path.join(cartesian_dir, self.lidar_dir)
            point_label_base_path = os.path.join(cartesian_dir, self.point_label_dir)
            voxel_base_path = os.path.join(cartesian_dir, self.occ_dir)
            scene_params = self.read_scene_params(os.path.join(voxel_base_path, 'params.json'))

            frame_ids = self.collect_frame_ids(image_base_path, lidar_base_path)
            if not frame_ids:
                continue

            sample_image_path = self.find_first_image(image_base_path, frame_ids[0])
            calib = self.build_topdown_calib(sample_image_path, scene_params['pc_range'], self.camera_height)
            P0 = calib['P0']
            T_velo_2_cam = calib['Tr']
            proj_matrix_0 = P0 @ T_velo_2_cam

            for frame_id in frame_ids:
                img_path = self.find_first_image(image_base_path, frame_id)
                lidar_path = os.path.join(lidar_base_path, frame_id + '.bin')
                lidar_label_path = os.path.join(point_label_base_path, frame_id + '.label')
                voxel_path = os.path.join(voxel_base_path, frame_id + self.occ_label_suffix)

                if not os.path.exists(img_path) or not os.path.exists(lidar_path):
                    continue
                if not os.path.exists(lidar_label_path):
                    lidar_label_path = None
                if not os.path.exists(voxel_path):
                    voxel_path = None

                scans.append(
                    {
                        'img_0_path': img_path,
                        'sequence': sequence,
                        'frame_id': frame_id,
                        'P0': P0,
                        'T_velo_2_cam': T_velo_2_cam,
                        'proj_matrix_0': proj_matrix_0,
                        'voxel_path': voxel_path,
                        'preprocessed_voxel_path': self.resolve_preprocessed_voxel_path(
                            sequence=sequence,
                            frame_id=frame_id,
                        ),
                        'lidar_path': lidar_path,
                        'lidar_label_path': lidar_label_path,
                        'source_occ_size': scene_params['occ_size'],
                        'source_pc_range': scene_params['pc_range'],
                        'scene_num_channels': scene_params['num_channels'],
                    }
                )
        return scans

    def get_ann_info(self, index, key='voxel_path'):
        info_dict = self.data_infos[index]
        preprocessed_info = info_dict.get('preprocessed_voxel_path')
        if self.prefer_preprocessed_labels and preprocessed_info is not None and os.path.exists(preprocessed_info):
            label = np.load(preprocessed_info)
            if self.num_classes is not None:
                label = label.copy()
                label[label >= self.num_classes] = self.ignore_label
            return label.astype(np.uint8)

        info = info_dict[key]
        if info is None:
            return None

        source_occ_size = np.array(info_dict.get('source_occ_size', self.occ_size), dtype=np.int32)
        source_pc_range = np.array(info_dict.get('source_pc_range', self.pc_range), dtype=np.float32)

        label = np.fromfile(info, dtype=np.uint32)
        source_num_voxels = int(np.prod(source_occ_size))
        if label.size != source_num_voxels:
            raise ValueError(
                f'Unexpected occupancy label length {label.size} for source occ_size {source_occ_size.tolist()}'
            )

        label = label.reshape(tuple(source_occ_size.tolist()))
        label = self.remap_occ_to_target(
            label,
            source_occ_size=source_occ_size,
            source_pc_range=source_pc_range,
            target_occ_size=self.target_occ_size,
            target_pc_range=self.target_pc_range,
        )

        if self.num_classes is not None:
            label = label.copy()
            label[label >= self.num_classes] = self.ignore_label

        return label.astype(np.uint8)

    def resolve_preprocessed_voxel_path(self, sequence, frame_id):
        if self.ann_file is None or self.ann_file == self.data_root:
            return None

        candidate = os.path.join(
            self.ann_file,
            self.preprocess_label_subdir,
            sequence,
            f'{frame_id}_{self.preprocess_scale}.npy',
        )
        return candidate if os.path.exists(candidate) else None

    @staticmethod
    def read_scene_params(params_path):
        if not os.path.exists(params_path):
            raise FileNotFoundError(f'Missing scene params file: {params_path}')

        with open(params_path, 'r') as f:
            params = json.load(f)

        occ_size = np.array([int(v) for v in params['grid_size']], dtype=np.int32)
        min_bound = np.array(params['min_bound'], dtype=np.float32)
        max_bound = np.array(params['max_bound'], dtype=np.float32)
        pc_range = np.concatenate([min_bound, max_bound]).astype(np.float32)

        return {
            'occ_size': occ_size,
            'pc_range': pc_range,
            'num_channels': int(params.get('num_channels', 0)),
            'coordinates': params.get('coordinates', 'cartesian'),
        }

    def remap_occ_to_target(
        self,
        label,
        source_occ_size,
        source_pc_range,
        target_occ_size,
        target_pc_range,
    ):
        if (
            np.array_equal(source_occ_size, target_occ_size)
            and np.allclose(source_pc_range, target_pc_range)
        ):
            return label

        target = np.zeros(tuple(target_occ_size.tolist()), dtype=np.uint32)

        source_min = source_pc_range[:3]
        source_max = source_pc_range[3:]
        target_min = target_pc_range[:3]
        target_max = target_pc_range[3:]

        source_steps = (source_max - source_min) / source_occ_size
        target_steps = (target_max - target_min) / target_occ_size

        occupied_coords = np.argwhere((label > 0) & (label != self.ignore_label))
        if occupied_coords.size == 0:
            return target

        occupied_labels = label[tuple(occupied_coords.T)]
        occupied_centers = source_min + (occupied_coords.astype(np.float32) + 0.5) * source_steps
        target_coords = np.floor((occupied_centers - target_min) / target_steps).astype(np.int32)

        valid_mask = np.all((target_coords >= 0) & (target_coords < target_occ_size), axis=1)
        if not np.any(valid_mask):
            return target

        target_coords = target_coords[valid_mask]
        occupied_labels = occupied_labels[valid_mask]

        target_flat = target.reshape(-1)
        flat_indices = np.ravel_multi_index(target_coords.T, tuple(target_occ_size.tolist()))

        if self.label_merge_mode == 'first':
            empty_mask = target_flat[flat_indices] == 0
            target_flat[flat_indices[empty_mask]] = occupied_labels[empty_mask]
        elif self.label_merge_mode == 'max':
            np.maximum.at(target_flat, flat_indices, occupied_labels)
        else:
            raise ValueError(f'Unsupported label_merge_mode: {self.label_merge_mode}')

        return target

    @staticmethod
    def collect_frame_ids(image_base_path, lidar_base_path):
        image_exts = ('*.png', '*.jpg', '*.jpeg')
        frame_ids = set()
        for pattern in image_exts:
            for image_path in glob.glob(os.path.join(image_base_path, pattern)):
                frame_ids.add(os.path.splitext(os.path.basename(image_path))[0])

        lidar_ids = {
            os.path.splitext(os.path.basename(path))[0]
            for path in glob.glob(os.path.join(lidar_base_path, '*.bin'))
        }
        return sorted(frame_ids & lidar_ids)

    @staticmethod
    def find_first_image(image_base_path, frame_id):
        for ext in ('.png', '.jpg', '.jpeg'):
            image_path = os.path.join(image_base_path, frame_id + ext)
            if os.path.exists(image_path):
                return image_path
        raise FileNotFoundError(f'Image for frame {frame_id} not found in {image_base_path}')

    @staticmethod
    def build_topdown_calib(image_path, pc_range, camera_height):
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size

        center_x = (pc_range[0] + pc_range[3]) / 2.0
        center_y = (pc_range[1] + pc_range[4]) / 2.0
        x_extent = max(abs(pc_range[0] - center_x), abs(pc_range[3] - center_x))
        y_extent = max(abs(pc_range[1] - center_y), abs(pc_range[4] - center_y))

        fx = (width * camera_height) / max(2.0 * y_extent, 1e-6)
        fy = (height * camera_height) / max(2.0 * x_extent, 1e-6)
        cx = width / 2.0
        cy = height / 2.0

        P0 = np.eye(4, dtype=np.float32)
        P0[:3, :4] = np.array(
            [
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        T_velo_2_cam = np.eye(4, dtype=np.float32)
        T_velo_2_cam[:3, :4] = np.array(
            [
                [0.0, 1.0, 0.0, -center_y],
                [-1.0, 0.0, 0.0, center_x],
                [0.0, 0.0, -1.0, camera_height],
            ],
            dtype=np.float32,
        )
        return {'P0': P0, 'Tr': T_velo_2_cam}

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
