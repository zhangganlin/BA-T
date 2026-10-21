# Agent Guide: BA-T

This guide is for coding agents (and people) working with this repository. It
gives an overview of what the code does, how to run it, and what to keep in
mind when changing it.

BA-T ("BA-T: An Iterative Transformer for Two-View Bundle Adjustment", NeurIPS
2026) predicts dense 3D point maps and relative camera poses from images. A
single BA-T layer, inspired by bundle adjustment, recursively refines image
tokens and relative camera tokens. The main use case is two views (an image
pair). The same model also refines more than two views jointly on a view graph.
This repository contains inference code only.

--------------------------------------------------------------------------------

## Quick Reference

```bash
# setup: see README.md (torch first, then requirements.txt, optionally curope)
# weights: pretrains/bat.pth (default) or pretrains/bat_paper.pth (paper results); links in README.md

python demo.py --images A.jpg B.jpg --output outputs/demo                 # two views
python demo_multiview.py --images path/to/folder --output outputs/demo_mv  # V >= 2 views
```

```python
from ba_t.data import load_views
from ba_t.inference import load_model, run_pairs, run_multiview

model = load_model('pretrains/bat.pth', device='cuda')
view_a, view_b = load_views(['A.jpg', 'B.jpg'])
out = run_pairs(model, view_a, [view_b])          # two views
mv = run_multiview(model, load_views('folder/'))  # multi-view
```

--------------------------------------------------------------------------------

## Code Map

```
demo.py                          two-view example: 2 images (A, B) or a 2-image folder ->
                                 prediction.npz + pointcloud.ply
demo_multiview.py                multi-view example: >= 2 images, automatic or --graph view graph,
                                 pose graph optimization (default) or --no_pgo
ba_t/data.py                     collect_image_paths, preprocess_image, load_image, ImageViews,
                                 collate_views, load_views, to_original_pixels, denormalize
ba_t/inference.py                load_model; run_pairs (independent (A, B) pairs);
                                 run_multiview (graph -> joint refinement -> global poses)
ba_t/graph.py                    view_similarity, build_view_graph, max_spanning_tree
ba_t/pose_graph.py               select_edges, init_poses, optimize_pose_graph
ba_t/models/bat_model.py         BAT: encode, decode (refinement on a view graph + heads), forward
ba_t/models/blocks/ba_blocks.py  BundleAdjustmentLayer and its building blocks
ba_t/models/blocks/vit_blocks.py ViT blocks; attention via xformers if installed, else torch SDPA
ba_t/models/heads/               point-map head (pts3d + conf), pose head (rotation via SVD,
                                 translation, confidence)
ba_t/models/pos_embed/           RoPE2D (curope CUDA kernel if built, else a torch implementation);
                                 curope/ = CUDA extension sources
ba_t/utils/                      cropping, image normalization, head wrapper
pretrains/                       bat.pth / bat_paper.pth go here (git-ignored)
media/                           README images
```

--------------------------------------------------------------------------------

## How the Model Works

-   **Encoder:** a ViT-L (patch 16, RoPE) encodes each 224x224 image into 196
    tokens.
-   **View graph:** each edge `(i, j)` gets a relative camera token. The token
    encodes `T_j<-i`, and `camera_inverse` of it encodes `T_i<-j`. Two views
    use the graph `[(0, 1)]`.
-   **Refinement:** one shared `BundleAdjustmentLayer` runs `K =
    num_refine_iter` times (default 8). Each iteration does three things:
    -   every view cross-attends to the tokens of its neighbours, each
        conditioned on the shared edge's camera token;
    -   self-attention runs within each view;
    -   every edge's camera token is updated from the correlation between its
        two views.
-   **Iterations:** the refinement is causal, so iteration k equals a run with
    `K=k`. `all_iters=True` returns all of them. Useful K is roughly 4 to 24;
    K=8 is a good default.
-   **Heads:** each view gets a point map `pts3d (N,H,W,3)` and a confidence
    `conf (N,H,W)`. Each edge gets both relative poses, `pose_ij` (`T_j<-i`)
    and `pose_ji` (`T_i<-j`), each with a confidence in (0, 1).

### APIs

-   **`BAT.forward(views, graph=None, num_refine_iter=8, all_iters=False)`**
    -   views are dicts with `img (N,3,224,224)` in [-1, 1] and `true_shape (N,2)`
        int (H, W) on the CPU.
    -   `graph=None` means all pairs.
    -   It returns `{'views': [{pts3d, conf}], 'edges': [{pose_ij, pose_ij_conf,
        pose_ji, pose_ji_conf}]}`.
    -   `encode` and `decode` are available separately, e.g. to reuse encoder
        features.
-   **`run_pairs(model, view_a, views_b)`** (two views)
    -   Image **A** is paired independently with each image **B**; A is
        encoded once.
    -   It returns `{'a': [...], 'b': [...]}`, one entry per image B, with
        `pts3d`, `conf`, `relative_pose` and `relative_pose_conf`.
    -   Use the names A and B in user-facing code and docs.
-   **`run_multiview(model, views, graph=None, ...)`** (batch size 1, one scene)
    1.  If no graph is given, it builds one from the encoder features
        (`ba_t/graph.py`):
        -   similarity is the fraction of mutual-nearest-neighbour patches
            between two views;
        -   pairs with similarity >= 0.3 are connected, with at most 7
            neighbours per view;
        -   a maximum spanning tree is always added, so all views are
            connected.
    2.  It refines all views jointly on the graph.
    3.  It computes global poses (`ba_t/pose_graph.py`):
        -   edges with pose confidence < 0.5 x the median are dropped (a
            max-confidence spanning tree is always kept);
        -   poses are chained along that tree;
        -   then pose graph optimization runs over the kept edges, robust and
            weighted by the squared confidence.
    4.  It returns `graph`, `similarity`, `pts3d`, `conf`, `pose_ij`, `pose_ji`,
        their confidences, `pose_edges` (the edges used for the poses) and
        `cam_to_world` (V,4,4), with view 0's camera as the world frame.

### Conventions

-   **Cameras:** OpenCV convention (x right, y down, z forward). Each point
    map is in its own view's camera frame.
-   **Relative poses:** `T_B<-A = inv(c2w_B) @ c2w_A` maps camera-A points
    into camera B. The two directions of an edge are predicted separately, so
    they are only approximately inverse to each other.
-   **Scale:** arbitrary and not metric. Within a pair, points and translation
    share one scale. Within a multi-view run, all views and edges share one
    scale.
-   **Confidences:** point confidence `conf` is >= 1, higher is better. Pose
    confidences are in (0, 1). The demos use pose confidence internally but
    don't print it.
-   **Graph size:** the model was trained on graphs of 2 to 8 views. Larger
    graphs work; the 7-neighbour cap keeps each view's neighbourhood in that
    range.

### Preprocessing (`ba_t/data.py`)

This matches the preprocessing the model was trained with, assuming the
principal point is at the image centre:

1.  EXIF transpose, then convert to RGB.
2.  Crop centred on the image centre.
3.  Lanczos-resize so that the short side becomes 224.
4.  Centre-crop to 224x224.
5.  Normalize to [-1, 1].

`view['crop']` records the transform, and `to_original_pixels` maps 224x224
pixels back to the original image. Only the central square is seen by the
model.

--------------------------------------------------------------------------------

## Rules for Changes

1.  **Keep the checkpoint loading.** Both released checkpoints,
    `pretrains/bat.pth` (default: further fine-tuned with more iterations and
    on multi-view graphs) and `pretrains/bat_paper.pth` (the paper's two-view
    checkpoint), are plain `state_dict`s of the same 405 tensors, loaded with
    `strict=True` and `weights_only=True`. Don't rename `nn.Module` attributes or parameters in
    `ba_t/models`. Some modules (`enc_norm`, `camera_encoder`) are unused at
    inference but must stay so that the weights load.
2.  **Keep the numerics.** Model outputs and preprocessing must not change
    unless that is the goal. To change them anyway, save outputs on a few
    image pairs and graphs before and after, and compare.
3.  **Keep the repo self-contained.** It must run from a fresh clone with the
    steps in `README.md`. Don't add machine-specific paths or new heavy
    dependencies; `ba_t` needs only torch, numpy and pillow.
4.  **Keep outputs out of the repo.** Demo outputs go to `outputs/`
    (git-ignored) or a temp dir. Never commit weights.
5.  **Lint:** `pyflakes ba_t demo.py demo_multiview.py` should be clean.

--------------------------------------------------------------------------------

## Environment

-   **Tested versions:** Python 3.11, torch 2.5.1 (CUDA 12.1), torchvision
    0.20.1, xformers 0.0.28.post3 (optional), numpy 2.x, pillow, open3d 0.20
    (only for viewing `.ply` files). Install torch from the CUDA index first,
    then `pip install -r requirements.txt` (or `pip install -e .`).
-   **curope** (optional CUDA RoPE kernel):
    -   build it with `cd ba_t/models/pos_embed/curope && pip install
        --no-build-isolation .`;
    -   it needs `nvcc` matching torch's CUDA version; without a system toolkit,
        run `conda install -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-cudart-dev`;
    -   it is compiled for sm_75 to sm_90 plus PTX; add a `-gencode` entry in
        `setup.py` for other GPUs;
    -   it installs as the top-level module `curope`; import torch before it;
    -   the build leaves `build/` and `*.egg-info` behind; don't commit them.
-   **Fallbacks:**
    -   without curope, a torch RoPE runs (differences at float-rounding
        level, about 1e-3 on points);
    -   without xformers, torch SDPA runs (identical results);
    -   the CPU works too, slowly.
-   **Speed and memory** (RTX 4070, curope and xformers, K=8):
    -   one pair: about 50 ms and 1.3 GB;
    -   multi-view: 8 views in 0.4 s and 1.6 GB, 64 views in 3 s and 3.3 GB.

--------------------------------------------------------------------------------

## Pitfalls

-   **cuRoPE2D works in place and needs a specific layout:** `(B,H,N,D)`
    tokens that are a view of a `(B,N,H,D)`-contiguous buffer, as q/k from the
    qkv reshape are. Clone the input if you still need it.
-   **The camera token uses RoPE position -1.** The torch RoPE fallback
    computes cos/sin from the positions directly so that negative positions
    work. Don't replace it with a lookup table.
-   **Importing `ba_t.models.bat_model` enables TF32** for CUDA matmuls (the
    model was trained with it).
-   **Keep `true_shape` as a CPU int tensor.** The head wrapper reads it on the
    host.
-   **Running scripts from another directory:** the demos work from any cwd.
    Other scripts need `pip install -e .` or `PYTHONPATH=<repo>`.
