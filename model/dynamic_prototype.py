"""
Dynamic prototype utilities for HMNet STAGE 2–3 (Point 3).

A: K-shot attention aggregation. B: 2-step prototype refinement.
C: both A and B. Init near identity for stable training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_arg(args, name, default=None):
    if args is not None and name in args:
        return args[name]
    return default


def baseline_kshot_aggregate(shot_fg_feats, shot_protos):
    """Original HMNet mean aggregation.

    Args:
        shot_fg_feats: [B, K, C, H, W] masked foreground features per shot.
        shot_protos: [B, K, C, 1, 1] per-shot prototypes.

    Returns:
        supp_feat [B, C, H, W], supp_pro [B, C, 1, 1], uniform weights [B, K].
    """
    B, K = shot_fg_feats.shape[:2]
    supp_feat = shot_fg_feats.mean(dim=1)
    supp_pro = shot_protos.mean(dim=1)
    weights = torch.full((B, K), 1.0 / K, device=shot_fg_feats.device, dtype=shot_fg_feats.dtype)
    return supp_feat, supp_pro, weights


class KShotAttentionAgg(nn.Module):
    """A: Learnable attention pooling over K support shots."""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.q_proj = nn.Linear(dim, hidden, bias=False)
        self.k_proj = nn.Linear(dim, hidden, bias=False)
        self.scale = hidden ** -0.5
        nn.init.zeros_(self.q_proj.weight)
        nn.init.zeros_(self.k_proj.weight)

    def forward(self, query_feat, shot_fg_feats, shot_protos):
        """
        Args:
            query_feat: [B, C, H, W]
            shot_fg_feats: [B, K, C, H, W]
            shot_protos: [B, K, C, 1, 1]
        Returns:
            supp_feat, supp_pro, attn_weights [B, K]
        """
        B, K, C, H, W = shot_fg_feats.shape
        q = F.adaptive_avg_pool2d(query_feat, 1).flatten(1)
        fg_pool = shot_fg_feats.mean(dim=(-2, -1))
        proto = shot_protos.squeeze(-1).squeeze(-1)
        k = 0.5 * (fg_pool + proto)

        q_h = self.q_proj(q).unsqueeze(1)
        k_h = self.k_proj(k)
        logits = (q_h * k_h).sum(dim=-1) * self.scale
        weights = F.softmax(logits, dim=1)

        w = weights.view(B, K, 1, 1, 1)
        supp_feat = (shot_fg_feats * w).sum(dim=1)
        supp_pro = (shot_protos * w.view(B, K, 1, 1, 1)).sum(dim=1)
        return supp_feat, supp_pro, weights


class PrototypeRefiner(nn.Module):
    """B: Two-step query + prior conditioned prototype update."""

    def __init__(self, dim, prior_ch=2, num_steps=2):
        super().__init__()
        self.num_steps = num_steps
        self.steps = nn.ModuleList()
        in_ch = dim * 2 + prior_ch
        for _ in range(num_steps):
            step = nn.Sequential(
                nn.Conv2d(in_ch, dim, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            )
            nn.init.zeros_(step[-1].weight)
            if step[-1].bias is not None:
                nn.init.zeros_(step[-1].bias)
            self.steps.append(step)

    def forward(self, supp_pro, query_feat, prior_mask):
        """
        Args:
            supp_pro: [B, C, H, W]
            query_feat: [B, C, H, W]
            prior_mask: [B, 2, H, W]
        """
        proto = supp_pro
        ctx = torch.cat([query_feat, prior_mask], dim=1)
        for step in self.steps:
            delta = step(torch.cat([proto, ctx], dim=1))
            proto = proto + delta
        return proto


def resolve_dynamic_proto_flags(args, shot):
    """Parse ablation flags from args / yaml / --opts."""
    mode = _get_arg(args, 'dynamic_proto_mode', None)
    if mode is not None:
        mode = str(mode).upper()
    if mode in {'A', 'KSHOT', 'KSHOT_ATTN'}:
        return True, False
    if mode in {'B', 'REFINE', 'PROTO_REFINE'}:
        return False, True
    if mode in {'C', 'AB', 'FULL', 'BOTH'}:
        return True, True
    if mode in {'NONE', 'OFF', '0', ''}:
        return False, False

    use_kshot = bool(_get_arg(args, 'proto_kshot_attn', False))
    use_refine = bool(_get_arg(args, 'proto_refine', False))
    if shot <= 1:
        use_kshot = False
    return use_kshot, use_refine


class DynamicPrototypeModule(nn.Module):
    """Optional dynamic prototype: A and/or B."""

    def __init__(
        self,
        dim,
        shot,
        use_kshot_attn=False,
        use_proto_refine=False,
        kshot_hidden=64,
        refine_steps=2,
        prior_ch=2,
    ):
        super().__init__()
        self.use_kshot_attn = bool(use_kshot_attn) and shot > 1
        self.use_proto_refine = bool(use_proto_refine)

        if self.use_kshot_attn:
            self.kshot_agg = KShotAttentionAgg(dim, hidden=kshot_hidden)
        else:
            self.kshot_agg = None

        if self.use_proto_refine:
            self.refiner = PrototypeRefiner(dim, prior_ch=prior_ch, num_steps=refine_steps)
        else:
            self.refiner = None

    @property
    def enabled(self):
        return self.use_kshot_attn or self.use_proto_refine

    def aggregate(self, query_feat, shot_fg_feats, shot_protos):
        if self.use_kshot_attn:
            return self.kshot_agg(query_feat, shot_fg_feats, shot_protos)
        return baseline_kshot_aggregate(shot_fg_feats, shot_protos)

    def refine(self, supp_pro, query_feat, prior_mask):
        if self.use_proto_refine:
            return self.refiner(supp_pro, query_feat, prior_mask)
        return supp_pro


def build_dynamic_prototype(args, dim, shot):
    use_kshot, use_refine = resolve_dynamic_proto_flags(args, shot)
    if not use_kshot and not use_refine:
        return None
    return DynamicPrototypeModule(
        dim=dim,
        shot=shot,
        use_kshot_attn=use_kshot,
        use_proto_refine=use_refine,
        kshot_hidden=int(_get_arg(args, 'proto_kshot_hidden', 64)),
        refine_steps=int(_get_arg(args, 'proto_refine_steps', 2)),
        prior_ch=2,
    )
