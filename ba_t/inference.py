"""Checkpoint loading, two-view (pair) inference and multi-view inference for BA-T."""
import torch

from .graph import DEFAULT_MAX_NEIGHBORS, DEFAULT_SIM_THRESH, build_view_graph, view_similarity
from .models import BAT
from .pose_graph import DEFAULT_MIN_REL_CONF, init_poses, optimize_pose_graph, select_edges


def load_model(ckpt_path, device='cuda', verbose=True):
    """Build ``BAT()`` and load its weights (a plain state_dict, e.g. ``pretrains/bat.pth``; strict)."""
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model = BAT()
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    if verbose:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'Loaded BAT ({n_params:.1f}M params) from {ckpt_path}')
    return model


def _to_device(view, device):
    # true_shape stays on the cpu, as in training
    return {'img': view['img'].to(device, non_blocking=True), 'true_shape': view['true_shape']}


def _to_cpu(x):
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu(v) for v in x]
    return x.float().cpu()


@torch.no_grad()
def run_pairs(model, view_a, views_b, num_refine_iter=8, all_iters=False):
    """Run BA-T on the independent image pairs (view_a, views_b[i]).

    Inputs are batched views (see ``ba_t.data.collate_views``); image A is encoded once.
    Returns, with tensors on the cpu, {'a': [per image B], 'b': [per image B]}, where entry i
    holds the predictions of image A ('a') and image B ('b') for the pair (A, views_b[i]):
      - 'pts3d' (N,H,W,3): point map in that image's own camera frame
      - 'conf' (N,H,W): point confidence (>= 1)
      - 'relative_pose' (N,4,4): 'a': T_B<-A (maps A-camera points into the B camera); 'b': T_A<-B
      - 'relative_pose_conf' (N,): pose confidence in (0, 1)
    With all_iters=True, a list of num_refine_iter such dicts (iteration 1..K).
    """
    device = next(model.parameters()).device
    view_a = _to_device(view_a, device)
    feat_a, pos_a = model.encode([view_a])
    results = None
    for view_b in views_b:
        view_b = _to_device(view_b, device)
        feat_b, pos_b = model.encode([view_b])
        out = model.decode(feat_a + feat_b, pos_a + pos_b, [view_a['true_shape'], view_b['true_shape']],
                           [(0, 1)], num_refine_iter=num_refine_iter, all_iters=all_iters)
        out = out if all_iters else [out]
        if results is None:
            results = [{'a': [], 'b': []} for _ in out]
        for res, o in zip(results, out):
            (va, vb), (e,) = o['views'], o['edges']
            res['a'].append(dict(va, relative_pose=e['pose_ij'], relative_pose_conf=e['pose_ij_conf']))
            res['b'].append(dict(vb, relative_pose=e['pose_ji'], relative_pose_conf=e['pose_ji_conf']))
    results = _to_cpu(results)
    return results if all_iters else results[-1]


@torch.no_grad()
def run_multiview(model, views, graph=None, num_refine_iter=8, sim_thresh=DEFAULT_SIM_THRESH,
                  max_neighbors=DEFAULT_MAX_NEIGHBORS, min_rel_conf=DEFAULT_MIN_REL_CONF, optimize_poses=True):
    """Run BA-T jointly on V >= 2 images of one scene and recover global camera poses.

    Args:
        views: list of V views with batch size 1 (e.g. ``ba_t.data.load_views``)
        graph: list of undirected edges (i, j) between covisible views. Default (None): built
            from the encoder features, see ``ba_t.graph`` (sim_thresh, max_neighbors).
            Two views always use [(0, 1)].
        num_refine_iter: number of recursive BA-layer iterations (K)
        min_rel_conf: for the camera poses, drop edges whose pose confidence is below this
            fraction of the graph's median (a max-confidence spanning tree is always kept);
            0 keeps all edges
        optimize_poses: refine the camera poses with pose graph optimization over the kept
            edges, weighted by the squared pose confidence (``ba_t.pose_graph``); if False, the
            relative poses are chained along the max-confidence spanning tree
    Returns (cpu tensors):
        'graph': list of E edges (i, j)
        'similarity': (V,V) view similarity used to build the graph (None if graph was given)
        'pts3d' (V,H,W,3), 'conf' (V,H,W): per-view point maps, each in its own camera frame
        'pose_ij' (E,4,4): T_j<-i for graph[e] = (i, j); 'pose_ji' (E,4,4): T_i<-j
        'pose_ij_conf', 'pose_ji_conf' (E,): pose confidences in (0, 1)
        'pose_edges': list of the edge indices used for the camera poses
        'cam_to_world' (V,4,4): camera-to-world poses, world = camera frame of view 0
    Point maps and translations share one arbitrary scale (not metric).
    """
    V = len(views)
    if V < 2:
        raise ValueError(f'multi-view inference needs at least 2 views, got {V}')
    if any(v['img'].shape[0] != 1 for v in views):
        raise ValueError('run_multiview expects views with batch size 1 (one scene)')
    device = next(model.parameters()).device
    views = [_to_device(v, device) for v in views]
    feats, poss = model.encode(views)

    similarity = None
    if graph is None:
        if V == 2:
            graph = [(0, 1)]
        else:
            similarity = view_similarity(feats)
            graph = build_view_graph(similarity, threshold=sim_thresh, max_neighbors=max_neighbors)
    graph = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in graph})

    out = model.decode(feats, poss, [v['true_shape'] for v in views], graph, num_refine_iter=num_refine_iter)
    pose_ij = torch.cat([e['pose_ij'] for e in out['edges']]).double().cpu()
    pose_ji = torch.cat([e['pose_ji'] for e in out['edges']]).double().cpu()
    conf_ij = torch.cat([e['pose_ij_conf'] for e in out['edges']]).double().cpu()
    conf_ji = torch.cat([e['pose_ji_conf'] for e in out['edges']]).double().cpu()

    # camera poses from the confident edges: on 7-Scenes, the pose confidence separates good from
    # wrong edges well, and confidence-squared weights lower the trajectory error by ~10%
    conf = 0.5 * (conf_ij + conf_ji)
    keep = select_edges(V, graph, conf, min_rel_conf) if min_rel_conf > 0 else list(range(len(graph)))
    g = [graph[e] for e in keep]
    cam_to_world = init_poses(V, g, pose_ij[keep], pose_ji[keep], weights=conf[keep])
    if optimize_poses:
        weights = torch.stack([conf_ij[keep], conf_ji[keep]], dim=-1) ** 2
        cam_to_world = optimize_pose_graph(cam_to_world, g, pose_ij[keep], pose_ji[keep], weights=weights)

    return {'graph': graph,
            'similarity': similarity,
            'pts3d': torch.cat([v['pts3d'] for v in out['views']]).float().cpu(),
            'conf': torch.cat([v['conf'] for v in out['views']]).float().cpu(),
            'pose_ij': pose_ij.float(),
            'pose_ji': pose_ji.float(),
            'pose_ij_conf': conf_ij.float(),
            'pose_ji_conf': conf_ji.float(),
            'pose_edges': keep,
            'cam_to_world': cam_to_world.float()}
