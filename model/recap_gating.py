"""
Similarity-gated adaptive Support Recap for HMNet Mix-Mamba.

Low support-query similarity  -> stronger / denser support re-injection (closer to original HMNet).
High similarity               -> sparser per-window recap to reduce redundant support tokens.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_support_query_similarity(query_feat, supp_feat, supp_pro, corr_query_mask):
    """Build a 4-D similarity vector per batch sample.

    Args:
        query_feat: [B, C, H, W] merged query features (post init_merge_query).
        supp_feat: [B, C, H, W] merged support features.
        supp_pro: [B, C, H, W] broadcast support prototype.
        corr_query_mask: [B, 2, H, W] prior maps from STAGE 3.

    Returns:
        [B, 4] tensor: (feat_sim, proto_sim, prior_fg, prior_diff).
    """
    q = F.adaptive_avg_pool2d(query_feat, 1).flatten(1)
    s = F.adaptive_avg_pool2d(supp_feat, 1).flatten(1)
    p = F.adaptive_avg_pool2d(supp_pro, 1).flatten(1)

    feat_sim = F.cosine_similarity(q, s, dim=1)
    proto_sim = F.cosine_similarity(q, p, dim=1)
    prior_fg = corr_query_mask[:, 0].mean(dim=[1, 2])
    prior_diff = corr_query_mask[:, 1].mean(dim=[1, 2])
    return torch.stack([feat_sim, proto_sim, prior_fg, prior_diff], dim=1)


def resize_window_weights(window_weights, num_windows):
    """Resize per-window recap weights to match current spatial partition."""
    if window_weights.shape[2] == num_windows:
        return window_weights
    # [B, 1, W, 1, 1] -> interpolate along window axis
    w = window_weights.squeeze(-1).squeeze(-1)  # [B, 1, W]
    w = F.interpolate(w, size=num_windows, mode='linear', align_corners=True)
    return w.unsqueeze(-1).unsqueeze(-1)


def apply_adaptive_recap(x_s, ratio, recap_gate=None):
    """Gated support recap: replaces fixed ``repeat(ratio**2)`` in Mix-Mamba.

    Args:
        x_s: [B, 2, 1, L_S, C] support sequence (row/col stacked).
        ratio: window partition factor (default 4 in HMNet).
        recap_gate: optional dict from :class:`RecapGateNetwork`.

    Returns:
        [B, 2, ratio^2, L_S, C] support tokens tiled across query windows.
    """
    num_windows = ratio ** 2
    x_s_rep = x_s.repeat(1, 1, num_windows, 1, 1).contiguous()
    if recap_gate is None:
        return x_s_rep

    B = x_s.shape[0]
    strength = recap_gate['recap_strength']
    if strength.dim() == 4:
        strength = strength.unsqueeze(-1)  # [B, 1, 1, 1] -> [B, 1, 1, 1, 1]
    else:
        strength = strength.reshape(B, 1, 1, 1, 1)
    window_weights = recap_gate['window_weights']
    window_weights = resize_window_weights(window_weights, num_windows)

    x_s_rep = x_s_rep * strength * window_weights
    return x_s_rep


def modulate_recap_gate_for_layer(recap_gate, layer_idx, num_mix_layers=4):
    """Optional layer-wise modulation for deeper Mix-Mamba blocks (error accumulation)."""
    if recap_gate is None or layer_idx % 2 == 0:
        return recap_gate

    mix_depth = layer_idx // 2  # 0, 1, 2, 3 for layers 1, 3, 5, 7
    if num_mix_layers <= 1:
        return recap_gate

    # Deeper mix layers: slightly attenuate recap unless similarity is already low.
    depth_scale = 0.92 ** mix_depth
    sim = recap_gate.get('similarity')
    if sim is not None:
        mismatch = (1.0 - sim[:, 0:1]).clamp(0.0, 1.0).view(-1, 1, 1, 1)
        depth_scale = depth_scale + (1.0 - depth_scale) * mismatch

    out = dict(recap_gate)
    out['recap_strength'] = recap_gate['recap_strength'] * depth_scale
    cross = recap_gate.get('cross_scale', None)
    if cross is not None:
        out['cross_scale'] = cross * depth_scale.view(-1, 1, 1)
    return out


class RecapGateNetwork(nn.Module):
    """Lightweight MLP: support-query similarity -> recap strength / window weights / cross scale."""

    def __init__(self, hidden_dim=32, num_windows=16, min_strength=0.75, max_strength=2.0):
        super().__init__()
        self.num_windows = num_windows
        self.min_strength = min_strength
        self.max_strength = max_strength

        self.gate_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.strength_head = nn.Linear(hidden_dim, 1)
        self.window_head = nn.Linear(hidden_dim, num_windows)
        self.period_head = nn.Linear(hidden_dim, 1)
        self.cross_head = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.strength_head.weight)
        nn.init.zeros_(self.strength_head.bias)
        nn.init.zeros_(self.window_head.weight)
        nn.init.zeros_(self.window_head.bias)
        nn.init.zeros_(self.period_head.weight)
        nn.init.zeros_(self.period_head.bias)
        nn.init.zeros_(self.cross_head.weight)
        nn.init.zeros_(self.cross_head.bias)

    def forward(self, query_feat, supp_feat, supp_pro, corr_query_mask):
        sim = compute_support_query_similarity(query_feat, supp_feat, supp_pro, corr_query_mask)
        h = self.gate_mlp(sim)

        # Low feat_sim -> larger mismatch -> stronger recap (defaults near original HMNet).
        mismatch = (1.0 - sim[:, 0:1]).clamp(0.0, 1.0)
        strength = torch.sigmoid(self.strength_head(h) + mismatch)
        recap_strength = self.min_strength + (self.max_strength - self.min_strength) * strength

        window_logits = self.window_head(h)
        window_weights = torch.sigmoid(window_logits + mismatch * 0.5)

        # Soft periodic recap: low similarity -> period ~ 1 (dense); high -> longer period (sparser).
        period = 1.0 + (self.num_windows - 1) * torch.sigmoid(self.period_head(h) + sim[:, 0:1] * 0.5)
        idx = torch.arange(self.num_windows, device=query_feat.device, dtype=query_feat.dtype)
        phase = 2.0 * math.pi * idx.view(1, -1) / period.clamp(min=1.0)
        periodic = torch.cos(phase) * 0.5 + 0.5
        window_weights = window_weights * periodic

        # Keep at least half recap everywhere for stability; init ~= all ones when sim low.
        window_weights = 0.5 + 0.5 * window_weights

        cross_scale = torch.sigmoid(self.cross_head(h) + mismatch * 0.5)
        cross_scale = 0.5 + 1.5 * cross_scale  # [0.5, 2.0]

        return {
            'recap_strength': recap_strength.view(-1, 1, 1, 1),
            'window_weights': window_weights.view(-1, 1, self.num_windows, 1, 1),
            'cross_scale': cross_scale.view(-1, 1, 1),
            'similarity': sim,
        }
