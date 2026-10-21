import torch
import torch.nn as nn
from functools import partial
from .vit_blocks import Attention, Mlp, DropPath, XFormer_Attention, CrossAttention


class CameraTokenEncoder(nn.Module):
    def __init__(self, d_model=768, fourier_dim=4):
        super().__init__()
        self.fourier_dim = fourier_dim
        
        in_dim = 8  # rotvec(3), trans(3), fx_fy(2)

        # expanded dim after Fourier features:
        feat_dim = in_dim * (1 + 2 * fourier_dim) if fourier_dim > 0 else in_dim

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, d_model)   # 768
        )

    def fourier(self, x):
        """x: (B, C)"""
        outs = [x]
        for i in range(self.fourier_dim):
            f = 2 ** i
            outs.append(torch.sin(f * x))
            outs.append(torch.cos(f * x))
        return torch.cat(outs, dim=-1)

    def so3_log(self,R):
        """
        R: (B, 3, 3) rotation matrices
        Returns: (B, 3) axis-angle (log SO(3))
        """

        # trace(R)
        cos_theta = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)

        theta = torch.acos(cos_theta)  # (B,)

        # compute axis
        wx = R[:, 2, 1] - R[:, 1, 2]
        wy = R[:, 0, 2] - R[:, 2, 0]
        wz = R[:, 1, 0] - R[:, 0, 1]
        w = torch.stack([wx, wy, wz], dim=-1)  # (B, 3)

        # For small angle, use first-order approximation
        small = theta < 1e-5

        # axis-angle magnitude = theta
        # axis = w / (2 sin θ)
        sin_theta = torch.sin(theta)

        # avoid division by zero
        scale = theta / (2 * sin_theta)
        scale = scale.unsqueeze(-1)  # (B,1)

        rotvec = scale * w  # (B, 3)

        # small-angle: rotvec ≈ w/2
        rotvec[small] = (0.5 * w[small])

        return rotvec

    def forward(self, T, fx_fy):
        """
        T: (B, 4, 4)
        fx_fy: (B, 2)
        Returns: (B, 768) camera token
        """

        # --- convert rotation to log SO(3) ---
        R = T[:, :3, :3]
        t = T[:, :3, 3]

        rotvec = self.so3_log(R)       # (B, 3)
        trans  = t                # (B, 3)

        vec = torch.cat([rotvec, trans, fx_fy], dim=-1)  # (B, 8)

        if self.fourier_dim > 0:
            vec = self.fourier(vec)  # (B, feat_dim)

        token = self.proj(vec)       # (B, 768)
        return token

class ModLN(nn.Module):
    """
    Modulation with adaLN.

    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """

    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1.0 + scale) + shift

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mlp(mod).chunk(2, dim=-1)  # [N,1,D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]

class ConditionModulationBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(ModLN, eps=1e-6),
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mod, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x, mod), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x, mod)))
        return x


class TokenCorrRes(nn.Module):
    
    def __init__(self, dim, rope=None, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(1*dim, dim)
        self.proj =  nn.Sequential(
            nn.Linear(1*dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rope = rope
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(self, tokens_a, tokens_b, qpos, kpos, return_attn_score=False):
        B, Na, C = tokens_a.shape
        Nb = tokens_b.shape[1]
        
        qa = self.projq(tokens_a).reshape(B, Na, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        va = self.projv(tokens_a).reshape(B, Na, C)

        kb = self.projk(tokens_b).reshape(B, Nb, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        vb = self.projv(tokens_b).reshape(B, Nb, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.rope is not None:
            qa = self.rope(qa, qpos) # (B, H, N, K)
            kb = self.rope(kb, kpos)
            
        attn_ab = (qa @ kb.transpose(-2, -1)) * self.scale
        attn_ab = attn_ab.softmax(dim=-1)
        attn_ab = self.attn_drop(attn_ab)
        
        va_in_b = (attn_ab @ vb).transpose(1, 2).reshape(B, Na, C)
        va = self.norm_a(va)
        va_in_b = self.norm_b(va_in_b)
        # x = torch.cat([va, va_in_b, (va - va_in_b).abs()], dim=-1)
        x = (va - va_in_b)#.abs()

        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attn_score:
            return x, attn_ab
        return x


class CrossAttnLayer(nn.Module):
    def __init__(self, dim, num_heads, use_mlp=False, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_y = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.use_mlp = use_mlp
        if use_mlp:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_input, y_input, pos_x, pos_y):
        y = self.norm_y(y_input)
        attn_result, attn_score = self.cross_attn(self.norm1(x_input), y, y, pos_x, pos_y, return_attn_score=True)
        x = x_input + self.drop_path(attn_result)
        if self.use_mlp:
            # x = x + self.drop_path(self.mlp(self.norm2(x)))
            x = self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_score

class BundleAdjustmentLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__()
        self.dec_embed_dim = dim
        self.norm_attn = norm_layer(dim)
        self.norm_attn2 = norm_layer(dim)
        self.norm_attn3 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn = XFormer_Attention(self.attn)

        self.attn2 = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = XFormer_Attention(self.attn2)

        self.attn3 = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn3 = XFormer_Attention(self.attn3)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.geo_feat_num_heads = 8
        # self.geo_feat_attn = CrossAttnLayer(dim, rope=rope, num_heads=self.geo_feat_num_heads, qkv_bias=qkv_bias, 
        #                                     attn_drop=attn_drop, norm_layer=norm_layer)
        self.num_cross_layers = 1
        self.num_cross_attn_heads = num_heads
        self.cross_attn_layers = nn.ModuleList([
            CrossAttnLayer(dim, self.num_cross_attn_heads, True, mlp_ratio, qkv_bias, drop, attn_drop, 
                           drop_path, act_layer, norm_layer, rope)
            for _ in range(self.num_cross_layers)
        ])

        # self.cross_attn_layers_fine = nn.ModuleList([
        #     CrossAttnLayer(dim, self.num_cross_attn_heads, True, mlp_ratio, qkv_bias, drop, attn_drop, 
        #                    drop_path, act_layer, norm_layer, rope)
        #     for _ in range(self.num_cross_layers)
        # ])

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.camera_inverse = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.,)
        
        self.token_transforms = nn.ModuleList([
            nn.ModuleList(
                [
                    ConditionModulationBlock(
                        self.dec_embed_dim,
                        num_heads=12,
                        mlp_ratio=4.0,
                        qkv_bias=True,
                        rope=rope,
                    )
                    for _ in range(1)
                ]
            ) for _ in range(self.num_cross_layers)
        ])
        self.camera_attn = CrossAttnLayer(dim, rope=rope, num_heads=12, qkv_bias=qkv_bias, 
                                          attn_drop=attn_drop, drop=drop, norm_layer=norm_layer,
                                          use_mlp=True, mlp_ratio=mlp_ratio, drop_path=drop_path, act_layer=act_layer)
        self.geo_norm = norm_layer(dim)
        self.cam_norm = norm_layer(dim)
        
        self.token_corr_res = TokenCorrRes(dim, rope=rope, num_heads=8, qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop)

    def _transform(self, g, t, pos):
        """Condition the tokens g of one view on a relative camera token t."""
        for trans in self.token_transforms[0]:
            g = trans(g, t, pos)
        return g

    def forward(self, gs, poss, ts, graph, update_cam_token=True):
        """One BA step on a view graph (two views = the graph [(0, 1)]).

        Args:
            gs: list of V view tokens (N, S, D)
            poss: list of V token positions (N, S, 2)
            ts: list of E camera tokens (N, 1, D); ts[e] belongs to edge graph[e] = (i, j)
                and encodes T_j<-i (camera_inverse(ts[e]) encodes T_i<-j)
            graph: list of E undirected edges (i, j)
        Returns:
            updated (gs, ts)
        """
        n_views = len(gs)
        N = gs[0].shape[0]
        ts_inv = list(torch.split(self.camera_inverse(torch.cat(ts, dim=0)), N, dim=0))

        # every view cross-attends to the tokens of all its neighbours, each neighbour
        # transformed with the camera token of the shared edge
        others = [[] for _ in range(n_views)]
        for e, (i, j) in enumerate(graph):
            others[i].append((self._transform(gs[j], ts_inv[e], poss[j]), poss[j]))
            others[j].append((self._transform(gs[i], ts[e], poss[i]), poss[i]))
        blk = self.cross_attn_layers[0]
        g_updates = []
        for v in range(n_views):
            g_others = torch.cat([g for g, _ in others[v]], dim=1)
            pos_others = torch.cat([p for _, p in others[v]], dim=1)
            g_updates.append(blk(gs[v], g_others, poss[v], pos_others)[0])

        # self-attention, batched over the views
        g_update = torch.cat(g_updates, dim=0)
        pos = torch.cat(poss, dim=0)
        g_self = self.drop_path(self.attn(self.norm_attn(g_update), pos))
        g_self = self.drop_path(self.attn2(self.norm_attn2(g_self), pos))
        g_self = self.drop_path(self.attn3(self.norm_attn3(g_self), pos))
        g_new = self.geo_norm(g_update + g_self + torch.cat(gs, dim=0))
        gs_new = list(torch.split(g_new, N, dim=0))

        if not update_cam_token:
            return gs_new, list(ts)

        # camera token update from the correlation residual of every edge, batched over the edges
        g_i = torch.cat([gs_new[i] for i, _ in graph], dim=0)
        g_j = torch.cat([self._transform(gs_new[j], ts_inv[e], poss[j]) for e, (_, j) in enumerate(graph)], dim=0)
        pos_i = torch.cat([poss[i] for i, _ in graph], dim=0)
        pos_j = torch.cat([poss[j] for _, j in graph], dim=0)
        res = self.token_corr_res(g_i, g_j, pos_i, pos_j)
        cam_pos = -torch.ones(pos_i.shape[0], 1, 2, device=pos_i.device, dtype=pos_i.dtype)
        t = torch.cat(ts, dim=0)
        t_delta, _ = self.camera_attn(t, res, cam_pos, pos_i)
        ts_new = list(torch.split(self.cam_norm(t + t_delta), N, dim=0))
        return gs_new, ts_new
        
