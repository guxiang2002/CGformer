import argparse
import json
import os
from pathlib import Path

import numpy as np


def get_mlab():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from mayavi import mlab

    mlab.options.offscreen = True
    return mlab


LABEL_COLORS = np.array(
    [
        [0, 0, 0, 255],          # 0  empty / unknown
        [128, 64, 128, 255],     # 1  road
        [244, 35, 232, 255],     # 2  sidewalk
        [70, 70, 70, 255],       # 3  building
        [102, 102, 156, 255],    # 4  wall
        [190, 153, 153, 255],    # 5  fence
        [153, 153, 153, 255],    # 6  pole
        [250, 170, 30, 255],     # 7  traffic light
        [220, 220, 0, 255],      # 8  traffic sign
        [107, 142, 35, 255],     # 9  vegetation
        [152, 251, 152, 255],    # 10 terrain
        [70, 130, 180, 255],     # 11 sky
        [220, 20, 60, 255],      # 12 pedestrian
        [255, 0, 0, 255],        # 13 rider
        [0, 0, 142, 255],        # 14 car
        [0, 0, 70, 255],         # 15 truck
        [0, 60, 100, 255],       # 16 bus
        [0, 80, 100, 255],       # 17 train
        [0, 0, 230, 255],        # 18 motorcycle
        [119, 11, 32, 255],      # 19 bicycle
        [110, 190, 160, 255],    # 20 static
        [170, 120, 50, 255],     # 21 dynamic
        [55, 90, 80, 255],       # 22 other
        [45, 60, 150, 255],      # 23 water
        [157, 234, 50, 255],     # 24 road lines
        [81, 0, 81, 255],        # 25 ground
        [150, 100, 100, 255],    # 26 bridge
        [230, 150, 140, 255],    # 27 rail track
        [180, 165, 180, 255],    # 28 guard rail
    ],
    dtype=np.uint8,
)

PEDESTRIAN_LABEL = 12
MODEL_GRID_SHAPE = (256, 256, 32)
MODEL_MIN_BOUND = np.array([-51.2, -51.2, -4.0], dtype=np.float32)
MODEL_MAX_BOUND = np.array([51.2, 51.2, 6.4], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Town10HD occupancy predictions.")
    parser.add_argument(
        "--data-root",
        default="/data/guxiang/gx/dataset/CarlaData/Town10HD",
        help="Town10HD dataset root.",
    )
    parser.add_argument(
        "--pred-root",
        default="/data/guxiang/gx/CGFormer/pred",
        help="Prediction save root from main.py --save_path, or a direct predictions folder.",
    )
    parser.add_argument("--split", default="Val", choices=["Train", "Val"])
    parser.add_argument("--scene", default="scene_08")
    parser.add_argument(
        "--output-root",
        default="/data/guxiang/gx/CGFormer/visualize_carla",
        help="Rendered output directory.",
    )
    parser.add_argument(
        "--source",
        default="pred",
        choices=["pred", "gt", "dataset_pred"],
        help="pred: model outputs, gt: evaluation labels, dataset_pred: dataset predictions/*.bin",
    )
    parser.add_argument("--frame", default=None, help="Optional single frame id, e.g. 000123.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--video-view", action="store_true")
    parser.add_argument("--ignore-label", type=int, default=255)
    return parser.parse_args()


def resolve_source_dir(args):
    scene = args.scene
    if args.source == "gt":
        return os.path.join(args.data_root, args.split, scene, "cartesian", "evaluation")
    if args.source == "dataset_pred":
        return os.path.join(args.data_root, args.split, scene, "cartesian", "predictions")

    candidates = [
        os.path.join(args.pred_root, "sequences", scene, "predictions"),
        os.path.join(args.pred_root, scene, "predictions"),
        args.pred_root,
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find prediction directory from {args.pred_root}")


def list_occ_files(source_dir, frame_id=None, frame_stride=1, max_frames=None):
    files = []
    for pattern in ("*.label", "*.bin"):
        files.extend(sorted(Path(source_dir).glob(pattern)))
    if frame_id is not None:
        files = [path for path in files if path.stem == frame_id]
    files = files[:: max(1, frame_stride)]
    if max_frames is not None:
        files = files[:max_frames]
    return files


def load_scene_params(eval_dir):
    params_path = os.path.join(eval_dir, "params.json")
    if not os.path.exists(params_path):
        return None
    with open(params_path, "r") as f:
        params = json.load(f)
    return {
        "grid_shape": tuple(int(v) for v in params["grid_size"]),
        "min_bound": np.asarray(params["min_bound"], dtype=np.float32),
        "max_bound": np.asarray(params["max_bound"], dtype=np.float32),
        "coordinates": params.get("coordinates", "cartesian"),
        "num_channels": int(params.get("num_channels", len(LABEL_COLORS))),
    }


def load_gt_voxels(occ_file, scene_params):
    if scene_params is None:
        raise FileNotFoundError(
            f"Missing params.json for GT visualization near {occ_file}."
        )
    voxels = np.fromfile(str(occ_file), dtype=np.uint32)
    expected_size = int(np.prod(scene_params["grid_shape"]))
    if voxels.size != expected_size:
        raise ValueError(
            f"GT file {occ_file} has {voxels.size} elements, expected {expected_size}."
        )
    return voxels.reshape(scene_params["grid_shape"]).astype(np.uint32), scene_params


def infer_spec(occ_file, scene_params, source):
    candidate_params = []
    if scene_params is not None:
        candidate_params.append(scene_params)
    candidate_params.append(
        {
            "grid_shape": MODEL_GRID_SHAPE,
            "min_bound": MODEL_MIN_BOUND,
            "max_bound": MODEL_MAX_BOUND,
            "coordinates": "cartesian",
            "num_channels": len(LABEL_COLORS),
        }
    )

    # main.py saves model predictions as uint16, while Town10HD GT labels are uint32.
    candidate_dtypes = [np.uint16, np.uint32] if source == "pred" else [np.uint32, np.uint16]

    for dtype in candidate_dtypes:
        label = np.fromfile(str(occ_file), dtype=dtype)
        for params in candidate_params:
            if label.size == int(np.prod(params["grid_shape"])):
                return label.reshape(params["grid_shape"]), params

    expected_sizes = [int(np.prod(params["grid_shape"])) for params in candidate_params]
    raise ValueError(
        f"Unsupported prediction size for {occ_file}. "
        f"Tried dtypes {[np.dtype(dtype).name for dtype in candidate_dtypes]} "
        f"and expected element counts {expected_sizes}."
    )


def load_counts(source, occ_file, grid_shape):
    count_path = Path(occ_file).with_suffix(".bin")
    if source in {"gt", "dataset_pred"} and count_path.exists():
        counts = np.fromfile(str(count_path), dtype=np.float32)
        if counts.size == int(np.prod(grid_shape)):
            return counts.reshape(grid_shape)
    return np.ones(grid_shape, dtype=np.float32)


def load_gt_counts(occ_file, scene_params):
    count_path = Path(occ_file).with_suffix(".bin")
    if not count_path.exists():
        raise FileNotFoundError(f"Missing GT count file: {count_path}")

    counts = np.fromfile(str(count_path), dtype=np.float32)
    expected_size = int(np.prod(scene_params["grid_shape"]))
    if counts.size != expected_size:
        raise ValueError(
            f"GT count file {count_path} has {counts.size} elements, expected {expected_size}."
        )
    return counts.reshape(scene_params["grid_shape"])


def get_grid_coords(dims, min_bound, max_bound, coordinates):
    dims = np.asarray(dims, dtype=np.int32)
    min_bound = np.asarray(min_bound, dtype=np.float32)
    max_bound = np.asarray(max_bound, dtype=np.float32)

    intervals = (max_bound - min_bound) / dims
    axes = [
        np.linspace(min_bound[i], max_bound[i], num=dims[i], endpoint=False) + intervals[i] / 2
        for i in range(3)
    ]
    xv, yv, zv = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    coords = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

    if coordinates == "cylindrical":
        rho = coords[:, 0]
        theta = coords[:, 1]
        z = coords[:, 2]
        coords = np.stack([rho * np.cos(theta), rho * np.sin(theta), z], axis=1)

    return coords, intervals


def configure_camera(scene, occupied_points, camera_offset_scale):
    mins = occupied_points.min(axis=0)
    maxs = occupied_points.max(axis=0)
    center = (mins + maxs) / 2.0
    extents = np.maximum(maxs - mins, 1e-3)
    z_extent = extents[2]
    max_extent = max(float(np.max(extents)), 20.0)

    focal_point = center.copy()
    focal_point[2] -= 0.15 * z_extent
    position = focal_point + np.asarray(camera_offset_scale, dtype=np.float32) * max_extent

    scene.camera.position = position.tolist()
    scene.camera.focal_point = focal_point.tolist()
    scene.camera.view_angle = 24.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [1.0, 8.0 * max_extent]
    scene.camera.compute_view_plane_normal()


def draw(voxels, counts, params, save_path, video_view):
    mlab = get_mlab()
    grid_coords, intervals = get_grid_coords(
        voxels.shape,
        params["min_bound"],
        params["max_bound"],
        params["coordinates"],
    )
    flat_labels = voxels.reshape(-1)
    flat_counts = counts.reshape(-1)

    valid_mask = flat_counts > 0
    occupied_mask = flat_labels > 0
    color_mask = flat_labels < len(LABEL_COLORS)
    mask = valid_mask & occupied_mask & color_mask

    if not np.any(mask):
        print("No occupied voxels found for", save_path)
        return

    occupied = np.hstack([grid_coords[mask], flat_labels[mask].reshape(-1, 1)])
    scale_factor = max(0.1, float(min(intervals[0], intervals[2]) * 0.9))

    figure = mlab.figure(size=(2200, 2200), bgcolor=(1, 1, 1))
    plot = mlab.points3d(
        occupied[:, 0],
        occupied[:, 1],
        occupied[:, 2],
        occupied[:, 3],
        mode="cube",
        scale_factor=scale_factor,
        opacity=1.0,
        vmin=0,
        vmax=len(LABEL_COLORS) - 1,
    )
    plot.glyph.scale_mode = "scale_by_vector"
    plot.module_manager.scalar_lut_manager.lut.table = LABEL_COLORS

    ped_mask = mask & (flat_labels == PEDESTRIAN_LABEL)
    if np.any(ped_mask):
        ped_coords = grid_coords[ped_mask]
        ped_scalars = flat_labels[ped_mask]
        ped_plot = mlab.points3d(
            ped_coords[:, 0],
            ped_coords[:, 1],
            ped_coords[:, 2],
            ped_scalars,
            mode="cube",
            scale_factor=scale_factor * 1.8,
            opacity=1.0,
            vmin=0,
            vmax=len(LABEL_COLORS) - 1,
        )
        ped_plot.glyph.scale_mode = "scale_by_vector"
        ped_plot.module_manager.scalar_lut_manager.lut.table = LABEL_COLORS

    camera_offset_scale = [1.25, -1.05, 1.55] if video_view else [1.1, -0.95, 1.7]
    configure_camera(figure.scene, occupied[:, :3], camera_offset_scale)
    figure.scene.render()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mlab.savefig(save_path)
    print("saved to", save_path)
    mlab.close(figure)


def main():
    args = parse_args()
    source_dir = resolve_source_dir(args)
    eval_dir = os.path.join(args.data_root, args.split, args.scene, "cartesian", "evaluation")
    scene_params = load_scene_params(eval_dir)

    occ_files = list_occ_files(source_dir, args.frame, args.frame_stride, args.max_frames)
    if not occ_files:
        raise FileNotFoundError(f"No occupancy files found in {source_dir}")

    output_dir = Path(args.output_root) / args.scene / args.source
    output_dir.mkdir(parents=True, exist_ok=True)

    for occ_file in occ_files:
        if args.source == "gt":
            voxels, params = load_gt_voxels(occ_file, scene_params)
            counts = load_gt_counts(occ_file, params)
        else:
            voxels, params = infer_spec(occ_file, scene_params, args.source)
            voxels = voxels.astype(np.uint32)
            counts = load_counts(args.source, occ_file, params["grid_shape"])

        voxels[voxels == args.ignore_label] = 0
        save_path = output_dir / f"{occ_file.stem}.png"
        draw(
            voxels=voxels,
            counts=counts,
            params=params,
            save_path=str(save_path),
            video_view=args.video_view,
        )


if __name__ == "__main__":
    main()
