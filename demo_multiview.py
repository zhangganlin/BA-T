"""BA-T multi-view example: joint refinement of V >= 2 images of one scene.

Examples:
    python demo_multiview.py --images path/to/folder --output outputs/demo_mv
    python demo_multiview.py --images img0.jpg img1.jpg img2.jpg img3.jpg --output outputs/demo_mv
    # explicit view graph instead of the automatic one
    python demo_multiview.py --images path/to/folder --graph 0-1 1-2 2-3 0-3

By default, the view graph (which views are covisible) is built from the encoder
features (see ba_t/graph.py), all views are refined jointly on that graph, and the
global camera poses are obtained by pose graph optimization over all edges.

The script writes to --output:
    prediction.npz   predictions (see below) + the 224x224 input images
    pointcloud.ply   coloured point cloud of all views in the camera frame of view 0

npz contents (V views, E edges, H = W = 224, OpenCV camera convention):
    paths          (V,)      source image paths (view order)
    graph          (E,2)     view graph edges (i, j)
    similarity     (V,V)     view similarity used to build the graph (only for the automatic graph)
    pts3d          (V,H,W,3) point maps, each in its own camera frame
    conf           (V,H,W)   point confidence (>= 1)
    pose_ij        (E,4,4)   T_j<-i for every edge (i, j): maps camera-i points into camera j
    pose_ji        (E,4,4)   T_i<-j
    pose_edges     (E',)     indices of the edges used for the camera poses (low-confidence edges dropped)
    cam_to_world   (V,4,4)   camera-to-world poses; world = camera frame of view 0
    imgs           (V,H,W,3) uint8 network input images
Point maps and translations share one arbitrary scale (not metric).
"""
import argparse
import os

import numpy as np
import torch

from ba_t.data import collect_image_paths, load_views, denormalize
from ba_t.graph import DEFAULT_MAX_NEIGHBORS, DEFAULT_SIM_THRESH
from ba_t.inference import load_model, run_multiview
from ba_t.pose_graph import DEFAULT_MIN_REL_CONF
from demo import rotation_angle_deg, save_ply, transform_points


def parse_edge(s):
    i, j = s.split('-')
    return int(i), int(j)


def get_args():
    parser = argparse.ArgumentParser('BA-T multi-view demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images', nargs='+', required=True,
                        help='image files and/or folders (natural sort order); at least two images')
    parser.add_argument('--ckpt', default='pretrains/bat.pth', help='model checkpoint')
    parser.add_argument('--num_refine_iter', type=int, default=8, help='number of BA-layer iterations K')
    parser.add_argument('--graph', nargs='+', type=parse_edge, default=None, metavar='I-J',
                        help='view graph edges as view-index pairs, e.g. 0-1 1-2 (default: automatic)')
    parser.add_argument('--sim_thresh', type=float, default=DEFAULT_SIM_THRESH,
                        help='automatic graph: connect views with similarity above this')
    parser.add_argument('--max_neighbors', type=int, default=DEFAULT_MAX_NEIGHBORS,
                        help='automatic graph: maximum number of neighbours per view')
    parser.add_argument('--min_rel_conf', type=float, default=DEFAULT_MIN_REL_CONF,
                        help='camera poses: ignore edges whose pose confidence is below this fraction of the '
                             'median (0: use all edges)')
    parser.add_argument('--no_pgo', action='store_true',
                        help='skip pose graph optimization (chain relative poses along a spanning tree)')
    parser.add_argument('--output', default='outputs/demo_mv', help='output folder')
    parser.add_argument('--conf_thresh', type=float, default=3.0,
                        help='only keep points with confidence above this in the .ply file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = get_args()
    paths = collect_image_paths(args.images)
    if len(paths) < 2:
        raise ValueError(f'expected at least 2 images, got {len(paths)}: {paths}')
    if args.graph is not None:
        bad = [e for e in args.graph if not (0 <= min(e) and max(e) < len(paths) and e[0] != e[1])]
        if bad:
            raise ValueError(f'invalid edges {bad} for {len(paths)} views (indices 0..{len(paths) - 1})')

    model = load_model(args.ckpt, device=args.device)
    views = load_views(paths)  # preprocessing identical to training: centre crop + resize to 224x224
    out = run_multiview(model, views, graph=args.graph, num_refine_iter=args.num_refine_iter,
                        sim_thresh=args.sim_thresh, max_neighbors=args.max_neighbors,
                        min_rel_conf=args.min_rel_conf, optimize_poses=not args.no_pgo)

    imgs = np.stack([denormalize(v['img'][0]) for v in views])
    pts3d, conf, c2w = out['pts3d'].numpy(), out['conf'].numpy(), out['cam_to_world'].numpy()
    os.makedirs(args.output, exist_ok=True)
    extra = {} if out['similarity'] is None else {'similarity': out['similarity'].numpy()}
    np.savez_compressed(
        os.path.join(args.output, 'prediction.npz'),
        paths=np.array(paths), graph=np.array(out['graph']), pts3d=pts3d, conf=conf,
        pose_ij=out['pose_ij'].numpy(), pose_ji=out['pose_ji'].numpy(), pose_edges=np.array(out['pose_edges']),
        cam_to_world=c2w, imgs=imgs, **extra)

    # all point maps in the camera frame of view 0
    keep = conf > args.conf_thresh
    pts = np.concatenate([transform_points(c2w[v], pts3d[v][keep[v]]) for v in range(len(paths))])
    cols = np.concatenate([imgs[v][keep[v]] for v in range(len(paths))])
    save_ply(os.path.join(args.output, 'pointcloud.ply'), pts, cols)

    np.set_printoptions(precision=4, suppress=True)
    for v, p in enumerate(paths):
        print(f'view {v}: {p}')
    print(f'K = {args.num_refine_iter} refinement iterations, '
          f'{"given" if args.graph is not None else "automatic"} view graph with {len(out["graph"])} edges:')
    print('  ' + ' '.join(f'{i}-{j}' for i, j in out['graph']))
    dropped = [out['graph'][e] for e in range(len(out['graph'])) if e not in set(out['pose_edges'])]
    if dropped:
        print(f'low-confidence edges not used for the camera poses: {" ".join(f"{i}-{j}" for i, j in dropped)}')
    print(f'camera poses in the camera frame of view 0 '
          f'({"spanning tree" if args.no_pgo else "pose graph optimization"}; arbitrary scale):')
    for v in range(len(paths)):
        print(f'  view {v}: rotation {rotation_angle_deg(c2w[v][:3, :3]):6.2f} deg, centre {c2w[v][:3, 3]}')
    print(f'saved {os.path.join(args.output, "prediction.npz")} and '
          f'{os.path.join(args.output, "pointcloud.ply")} ({len(pts)} points)')


if __name__ == '__main__':
    main()
