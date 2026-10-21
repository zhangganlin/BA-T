"""BA-T example: relative pose + point maps for an image pair (A, B).

Examples:
    python demo.py --images path/to/A.jpg path/to/B.jpg --output outputs/demo
    # a folder with exactly two images (natural sort order: the first one is A)
    python demo.py --images path/to/folder --output outputs/demo

The script prints the predicted relative pose and writes to --output:
    prediction.npz   predictions (see below) + the 224x224 input images
    pointcloud.ply   coloured point cloud of both images in the camera frame of A

npz contents (H = W = 224, OpenCV camera convention: x right, y down, z forward):
    pts3d_a / pts3d_b         (H,W,3) point maps, each in its own camera frame
    conf_a / conf_b           (H,W)   point confidence (>= 1)
    T_b_from_a                (4,4)   maps camera-A points into camera B
    T_a_from_b                (4,4)   maps camera-B points into camera A
    img_a / img_b             (H,W,3) uint8 network input images
    path_a / path_b                   source image paths
Point maps and translations share one arbitrary scale (not metric).
"""
import argparse
import os

import numpy as np
import torch

from ba_t.data import collect_image_paths, load_views, denormalize
from ba_t.inference import load_model, run_pairs


def get_args():
    parser = argparse.ArgumentParser('BA-T demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images', nargs='+', required=True,
                        help='two image files (A B), or a folder containing exactly two images')
    parser.add_argument('--ckpt', default='pretrains/bat.pth', help='model checkpoint')
    parser.add_argument('--num_refine_iter', type=int, default=8, help='number of BA-layer iterations K')
    parser.add_argument('--output', default='outputs/demo', help='output folder')
    parser.add_argument('--conf_thresh', type=float, default=3.0,
                        help='only keep points with confidence above this in the .ply file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def transform_points(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def save_ply(path, pts, colors):
    """Binary little-endian PLY with xyz float32 + rgb uint8 (no extra dependency)."""
    vertex = np.empty(len(pts), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex['x'], vertex['y'], vertex['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    vertex['red'], vertex['green'], vertex['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(pts)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertex.tobytes())


def main():
    args = get_args()
    paths = collect_image_paths(args.images)
    if len(paths) != 2:
        raise ValueError(f'expected exactly 2 images (A and B), got {len(paths)}: {paths}')

    model = load_model(args.ckpt, device=args.device)
    view_a, view_b = load_views(paths)  # preprocessing identical to training: centre crop + resize to 224x224

    out = run_pairs(model, view_a, [view_b], num_refine_iter=args.num_refine_iter)
    a, b = out['a'][0], out['b'][0]
    T_ba = a['relative_pose'][0].numpy()   # A -> B
    T_ab = b['relative_pose'][0].numpy()   # B -> A
    pts_a, pts_b = a['pts3d'][0].numpy(), b['pts3d'][0].numpy()
    conf_a, conf_b = a['conf'][0].numpy(), b['conf'][0].numpy()
    img_a, img_b = denormalize(view_a['img'][0]), denormalize(view_b['img'][0])

    os.makedirs(args.output, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.output, 'prediction.npz'),
        path_a=paths[0], path_b=paths[1],
        pts3d_a=pts_a, pts3d_b=pts_b, conf_a=conf_a, conf_b=conf_b,
        T_b_from_a=T_ba, T_a_from_b=T_ab,
        img_a=img_a, img_b=img_b)

    # both point maps in the camera frame of A
    keep_a = conf_a > args.conf_thresh
    keep_b = conf_b > args.conf_thresh
    pts = np.concatenate([pts_a[keep_a], transform_points(T_ab, pts_b[keep_b])])
    cols = np.concatenate([img_a[keep_a], img_b[keep_b]])
    save_ply(os.path.join(args.output, 'pointcloud.ply'), pts, cols)

    # camera centre of B in the frame of A, as a unit direction (the scale is arbitrary)
    t = T_ab[:3, 3] / max(np.linalg.norm(T_ab[:3, 3]), 1e-12)
    np.set_printoptions(precision=4, suppress=True)
    print(f'image A: {paths[0]}')
    print(f'image B: {paths[1]}')
    print(f'K = {args.num_refine_iter} refinement iterations')
    print(f'relative rotation        : {rotation_angle_deg(T_ba[:3, :3]):.2f} deg')
    print(f'camera B centre direction: {t}  (unit vector in the camera frame of A)')
    print(f'T_b_from_a (camera A -> camera B):\n{T_ba}')
    print(f'saved {os.path.join(args.output, "prediction.npz")} and '
          f'{os.path.join(args.output, "pointcloud.ply")} ({len(pts)} points)')


if __name__ == '__main__':
    main()
