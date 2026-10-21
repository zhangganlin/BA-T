"""View graph for multi-view inference: which views are covisible.

The default graph comes from the encoder features that BA-T computes anyway:
``view_similarity`` is the fraction of mutual nearest-neighbour patches between two
views (cosine similarity of the final encoder tokens), and ``build_view_graph``
connects every pair above a threshold, plus a maximum spanning tree so that all
views end up in one connected component.
"""
import torch

DEFAULT_SIM_THRESH = 0.3  # on 7-Scenes: ~0.94 of the kept pairs are covisible, ~0.83 of the covisible pairs are kept
DEFAULT_MAX_NEIGHBORS = 7  # the model was trained on graphs of up to 8 views (<= 7 neighbours per view)


@torch.no_grad()
def view_similarity(feats):
    """Pairwise view similarity in [0, 1] from encoder tokens.

    Args:
        feats: list of V encoder token tensors (1, S, D) or (S, D) (``BAT.encode``)
    Returns:
        (V, V) float tensor on the cpu; entry (i, j) is the fraction of patches of view i
        whose nearest neighbour in view j has view i's patch as its own nearest neighbour.
        The diagonal is 1.
    """
    F = torch.stack([f.reshape(-1, f.shape[-1]) for f in feats]).float()
    F = torch.nn.functional.normalize(F, dim=-1)
    V, S, _ = F.shape
    sim = torch.eye(V)
    idx = torch.arange(S, device=F.device)
    for i in range(V - 1):
        dots = torch.einsum('sd,jtd->jst', F[i], F[i + 1:])  # (V-i-1, S, S)
        nn_ij = dots.argmax(dim=2)  # nearest patch in view j of every patch of view i
        nn_ji = dots.argmax(dim=1)  # nearest patch in view i of every patch of view j
        mutual = torch.gather(nn_ji, 1, nn_ij) == idx
        sim[i, i + 1:] = sim[i + 1:, i] = mutual.float().mean(dim=1).cpu()
    return sim


def is_connected(n_views, graph):
    adj = [[] for _ in range(n_views)]
    for i, j in graph:
        adj[i].append(j)
        adj[j].append(i)
    seen, stack = {0}, [0]
    while stack:
        for w in adj[stack.pop()]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == n_views


def max_spanning_tree(n_views, edges, scores):
    """Edges of a maximum spanning tree (Kruskal) of the graph ``edges`` with edge ``scores``."""
    parent = list(range(n_views))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = set()
    for e in sorted(range(len(edges)), key=lambda e: -float(scores[e])):
        i, j = edges[e]
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.add((i, j))
    return tree


def build_view_graph(sim, threshold=DEFAULT_SIM_THRESH, max_neighbors=DEFAULT_MAX_NEIGHBORS):
    """Connected view graph from a (V, V) similarity matrix.

    1. Maximum spanning tree of the similarities: guarantees one connected component
       (each tree edge is the most similar link between two parts of the graph).
    2. Every other pair with similarity >= threshold, most similar first, as long as both
       views have fewer than max_neighbors neighbours (None: no limit).

    Returns:
        sorted list of edges (i, j) with i < j
    """
    sim = torch.as_tensor(sim, dtype=torch.float64)
    V = sim.shape[0]
    pairs = [(i, j) for i in range(V) for j in range(i + 1, V)]
    scores = [float(sim[i, j]) for i, j in pairs]
    edges = max_spanning_tree(V, pairs, scores)

    degree = [0] * V
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    for s, (i, j) in sorted(zip(scores, pairs), reverse=True):
        if s < threshold:
            break
        if (i, j) in edges:
            continue
        if max_neighbors is not None and (degree[i] >= max_neighbors or degree[j] >= max_neighbors):
            continue
        edges.add((i, j))
        degree[i] += 1
        degree[j] += 1
    return sorted(edges)
