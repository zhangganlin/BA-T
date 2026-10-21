"""Global camera poses from the relative poses of a view graph.

``select_edges`` drops edges with a low predicted pose confidence (keeping the graph
connected); ``init_poses`` chains the relative poses along the most confident spanning
tree; ``optimize_pose_graph`` then refines all camera poses jointly against every kept
edge (both predicted directions) with a robust, confidence-weighted loss. Poses are
camera-to-world 4x4 matrices with the world frame equal to the camera frame of view 0
(OpenCV convention).
"""
import torch

from .graph import max_spanning_tree

DEFAULT_MIN_REL_CONF = 0.5  # drop edges below this fraction of the graph's median pose confidence


def select_edges(n_views, graph, conf, min_rel_conf=DEFAULT_MIN_REL_CONF):
    """Indices of the edges to use for pose estimation.

    Keeps every edge whose pose confidence is >= min_rel_conf * (median confidence of the
    graph), plus a maximum-confidence spanning tree so that all views stay connected.
    On 7-Scenes, the pose confidence separates good (< 5 deg error) from bad (> 15 deg)
    edges with AUC 0.95.
    """
    conf = torch.as_tensor(conf, dtype=torch.float64).reshape(-1)
    tree = max_spanning_tree(n_views, graph, conf)
    thresh = min_rel_conf * conf.median()
    return [e for e, edge in enumerate(graph) if tuple(edge) in tree or conf[e] >= thresh]


def _skew(w):
    O = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([O, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], O, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], O], -1)], -2)


def _so3_exp(w):
    return torch.linalg.matrix_exp(_skew(w))


def _project_rotation(R):
    U, _, Vh = torch.linalg.svd(R)
    d = torch.sign(torch.det(U @ Vh))
    return U @ torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1)) @ Vh


def _edge_measurements(graph, pose_ij, pose_ji):
    """Both predicted directions as measurements of T_j<-i, as (2E,) lists of (i, j) and (2E,4,4)."""
    pairs = list(graph) + list(graph)
    T = torch.cat([pose_ij, torch.linalg.inv(pose_ji)], dim=0)
    return pairs, T


def init_poses(n_views, graph, pose_ij, pose_ji, weights=None):
    """Camera-to-world poses by chaining relative poses from view 0 along a spanning tree.

    The tree is the maximum spanning tree of ``weights`` (e.g. the pose confidences;
    default: the first edges in graph order). For every tree edge, T_j<-i is the average of
    pose_ij and inv(pose_ji).
    """
    pose_ij, pose_ji = pose_ij.double(), pose_ji.double()
    T_avg = torch.linalg.inv(pose_ji)
    T_avg[:, :3, 3] = 0.5 * (pose_ij[:, :3, 3] + T_avg[:, :3, 3])
    T_avg[:, :3, :3] = _project_rotation(pose_ij[:, :3, :3] + T_avg[:, :3, :3])
    if weights is None:
        weights = -torch.arange(len(graph), dtype=torch.float64)
    tree = max_spanning_tree(n_views, graph, weights)
    tree_edges = [e for e, edge in enumerate(graph) if tuple(edge) in tree]

    c2w = [None] * n_views
    c2w[0] = torch.eye(4, dtype=torch.float64)
    while any(c is None for c in c2w):
        for e in tree_edges:
            i, j = graph[e]
            if (c2w[i] is None) == (c2w[j] is None):
                continue
            if c2w[j] is None:   # c2w_j = c2w_i @ T_i<-j
                c2w[j] = c2w[i] @ torch.linalg.inv(T_avg[e])
            else:                # c2w_i = c2w_j @ T_j<-i
                c2w[i] = c2w[j] @ T_avg[e]
            break
        else:
            raise ValueError(f'view graph {graph} does not connect all {n_views} views')
    return torch.stack(c2w)


def optimize_pose_graph(c2w_init, graph, pose_ij, pose_ji, num_iters=100, robust_scale=0.3, trans_weight=1.0,
                        weights=None):
    """Refine camera-to-world poses so that they agree with all relative pose predictions.

    Minimizes, over the poses of views 1..V-1 (view 0 stays the identity), a weighted,
    Cauchy-robust sum over both predicted directions of every edge of
        ||R_err - I||_F^2 + trans_weight * ||t_err||^2 / s^2,
    where (R_err, t_err) = T_meas^-1 @ T_j<-i(poses) and s is the median predicted translation
    length (the scale of the predictions is arbitrary).

    Args:
        c2w_init: (V,4,4) initial camera-to-world poses (e.g. ``init_poses``)
        graph: list of E edges (i, j)
        pose_ij, pose_ji: (E,4,4) predicted T_j<-i and T_i<-j
        weights: optional (E,2) per-edge weights of the two directions (pose_ij, pose_ji),
            e.g. the squared pose confidences; default: uniform
    Returns:
        (V,4,4) float64 camera-to-world poses
    """
    pairs, T_meas = _edge_measurements(graph, pose_ij.double(), pose_ji.double())
    I_idx = torch.tensor([i for i, _ in pairs])
    J_idx = torch.tensor([j for _, j in pairs])
    T_meas_inv = torch.linalg.inv(T_meas)
    w = torch.ones(len(pairs), dtype=torch.float64) if weights is None else \
        torch.as_tensor(weights, dtype=torch.float64).reshape(-1, 2).T.reshape(-1)
    w = w / w.mean()
    scale = T_meas[:, :3, 3].norm(dim=-1).median().clamp(min=1e-8)

    c2w_init = c2w_init.double()
    V = c2w_init.shape[0]
    rot = torch.zeros(V - 1, 3, dtype=torch.float64, requires_grad=True)
    trans = torch.zeros(V - 1, 3, dtype=torch.float64, requires_grad=True)

    def poses():
        R = torch.cat([c2w_init[:1, :3, :3], _so3_exp(rot) @ c2w_init[1:, :3, :3]])
        t = torch.cat([c2w_init[:1, :3, 3], c2w_init[1:, :3, 3] + trans * scale])
        return R, t

    def loss_fn():
        R, t = poses()
        # T_j<-i = inv(c2w_j) @ c2w_i
        R_ji = R[J_idx].transpose(-1, -2) @ R[I_idx]
        t_ji = (R[J_idx].transpose(-1, -2) @ (t[I_idx] - t[J_idx]).unsqueeze(-1)).squeeze(-1)
        R_err = T_meas_inv[:, :3, :3] @ R_ji
        t_err = (T_meas_inv[:, :3, :3] @ t_ji.unsqueeze(-1)).squeeze(-1) + T_meas_inv[:, :3, 3]
        r2 = ((R_err - torch.eye(3, dtype=R_err.dtype)) ** 2).sum((-1, -2)) \
            + trans_weight * (t_err ** 2).sum(-1) / scale ** 2
        return (w * robust_scale ** 2 * torch.log1p(r2 / robust_scale ** 2)).sum()

    if V > 1:
        opt = torch.optim.LBFGS([rot, trans], lr=1, max_iter=num_iters, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-10, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss
        opt.step(closure)

    with torch.no_grad():
        R, t = poses()
        c2w = torch.eye(4, dtype=torch.float64).repeat(V, 1, 1)
        c2w[:, :3, :3] = R
        c2w[:, :3, 3] = t
    return c2w
