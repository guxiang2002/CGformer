_base_ = ['./4gpu-carla-cam.py']


data = dict(
    val=dict(
        split='train',
        scene_names=['scene_00'],
        test_mode=True,
    ),
    test=dict(
        split='train',
        scene_names=['scene_00'],
        test_mode=True,
    ),
)
