# --------------------------------------------------------
# BA-T: iterative transformer for two-view bundle adjustment.
# The views of a view graph (two views: the graph [(0, 1)]) are refined
# jointly and recursively by a single shared BundleAdjustmentLayer for
# `num_refine_iter` iterations.
# --------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True  # the released model was trained with TF32 (gpu >= Ampere)

from .pos_embed import RoPE2D
from .patch_embed import get_patch_embed

from .blocks.vit_blocks import Block
from .blocks.ba_blocks import BundleAdjustmentLayer, CameraTokenEncoder

from .heads import head_factory
from .heads.pose_head import PoseHead_small

from ..utils.misc import transpose_to_landscape

inf = float('inf')


class BAT(nn.Module):
    """BA-T model.

    - ViT-L image encoder (shared by all views, RoPE positional embedding)
    - recursive BundleAdjustmentLayer that jointly refines the tokens of all views and
      one relative camera token per edge of the view graph (two views: one edge)
    - heads for per-view 3D points + confidence and per-edge relative poses + confidence

    Do not rename module attributes: the state_dict keys of the released checkpoint
    depend on them.
    """
    def __init__(self,
                 img_size=(224,224),     # input image size
                 patch_size=16,          # patch_size
                 enc_embed_dim=1024,     # encoder feature dimension
                 enc_depth=24,           # encoder depth
                 enc_num_heads=16,       # encoder number of heads in the transformer block
                 dec_embed_dim=768,      # BA layer feature dimension
                 dec_num_heads=12,       # BA layer number of heads
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 pos_embed='RoPE100',    # positional embedding (RoPE with base frequency 100)
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                ):

        super().__init__()

        self.img_size = img_size

        # patch embeddings
        self.patch_size = patch_size
        self.patch_embed = get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim)
        # positional embedding (RoPE inside every attention block)
        assert pos_embed.startswith('RoPE'), f'Unknown pos_embed {pos_embed}'
        self.pos_embed = pos_embed
        self.rope = RoPE2D(freq=float(pos_embed[len('RoPE'):]))
        # transformer encoder
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)  # unused (features are not normalized), kept for the checkpoint
        # recursive bundle-adjustment layer
        self.dec_embed_dim = dec_embed_dim
        self.appearance_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.ba_layer = BundleAdjustmentLayer(dec_embed_dim, num_heads=dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                                              drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                                              norm_layer=norm_layer, rope=self.rope)
        # prediction heads
        self._set_downstream_head(output_mode, head_type, landscape_only,
                                  depth_mode, conf_mode, patch_size, img_size)

        # encoder of a known relative pose into a camera token; not used at inference,
        # kept so that the checkpoint loads with strict=True
        self.camera_encoder = CameraTokenEncoder(d_model=dec_embed_dim)

        self.update_cam_token = True

    def _set_downstream_head(self, output_mode, head_type, landscape_only,
                             depth_mode, conf_mode, patch_size, img_size):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.downstream_head_pts = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head_pts = transpose_to_landscape(self.downstream_head_pts, activate=landscape_only)
        self.head_pose_s = PoseHead_small(input_dim=self.dec_embed_dim)
        self.init_pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            )

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        for blk in self.enc_blocks:
            x = blk(x, pos)
        # NOTE: enc_norm is intentionally not applied (this is how the model was trained)
        return x, pos

    def encode(self, views):
        """Encode a list of views; returns lists of encoder tokens (N, S, 1024) and positions (N, S, 2)."""
        feats, poss = zip(*[self._encode_image(v['img'], v['true_shape']) for v in views])
        return list(feats), list(poss)

    def decode(self, feats, poss, true_shapes, graph, num_refine_iter=8, all_iters=False):
        """Refine encoded views jointly on a view graph and run the heads.

        Args:
            feats, poss: encoder tokens / positions of the V views (see ``encode``)
            true_shapes: list of V (N, 2) int tensors (H, W), on the cpu
            graph: list of E undirected edges (i, j) between view indices; must be connected
            num_refine_iter: number of recursive BA-layer iterations (K)
            all_iters: if True, return the predictions of every iteration 1..K
        Returns:
            {'views': [per view], 'edges': [per edge]}:
              - views[v]['pts3d'] (N,H,W,3): point map of view v in its own camera frame
              - views[v]['conf'] (N,H,W): point confidence (>= 1)
              - edges[e]['pose_ij'] (N,4,4): T_j<-i for graph[e] = (i, j)
                (maps camera-i points into camera j); edges[e]['pose_ji']: T_i<-j
              - edges[e]['pose_ij_conf'], ['pose_ji_conf'] (N,): pose confidence in (0, 1)
            With all_iters=True, a list of num_refine_iter such dicts (iteration 1..K).
        """
        graph = [tuple(int(v) for v in e) for e in graph]
        assert len(graph) > 0 and all(i != j for i, j in graph), f'invalid view graph {graph}'
        gs = [self.appearance_embed(f) for f in feats]
        ts = [self.init_pose_token.expand(feats[0].shape[0], -1, -1) for _ in graph]

        results = []
        for k in range(num_refine_iter):
            gs, ts = self.ba_layer(gs, poss, ts, graph, update_cam_token=self.update_cam_token)
            if all_iters or k == num_refine_iter - 1:
                results.append(self._heads(feats, gs, true_shapes, ts))
        return results if all_iters else results[0]

    def _heads(self, feats, gs, true_shapes, ts):
        views = []
        for feat, g, true_shape in zip(feats, gs, true_shapes):
            pts = self.head_pts([feat, g.float()], true_shape)
            views.append({'pts3d': pts['pts3d'], 'conf': pts['conf']})
        edges = []
        for t in ts:
            pose_ij = self.head_pose_s(t[:, 0, :].float())
            pose_ji = self.head_pose_s(self.ba_layer.camera_inverse(t)[:, 0, :].float())
            edges.append({'pose_ij': pose_ij['pose'], 'pose_ij_conf': pose_ij['conf'],
                          'pose_ji': pose_ji['pose'], 'pose_ji_conf': pose_ji['conf']})
        return {'views': views, 'edges': edges}

    def forward(self, views, graph=None, num_refine_iter=8, all_iters=False):
        """Predict point maps and relative poses for views connected by a view graph.

        Args:
            views: list of V >= 2 dicts with 'img' (N,3,H,W) normalized to [-1,1] and
                'true_shape' (N,2) int (H,W)
            graph: list of undirected edges (i, j) between covisible views; default: all pairs.
                Two views: [(0, 1)].
            num_refine_iter, all_iters: see ``decode``
        Returns:
            see ``decode``
        """
        if graph is None:
            graph = [(i, j) for i in range(len(views)) for j in range(i + 1, len(views))]
        feats, poss = self.encode(views)
        return self.decode(feats, poss, [v['true_shape'] for v in views], graph,
                           num_refine_iter=num_refine_iter, all_iters=all_iters)
