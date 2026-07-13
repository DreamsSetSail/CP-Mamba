"""
Lightweight Mamba error-reset (Point 5): pull query back toward early snapshots.

Mechanisms (optional, combinable):
  query_snapshot  — after each block: q += α_l · q_init
  support_skip    — after Mix blocks (odd idx): q += β_l · s_init

Gates init small (sigmoid ≈ 0.12) so training starts near original HMNet.
"""
import torch
import torch.nn as nn


def _get_arg(args, name, default=None):
    if args is not None and name in args:
        return args[name]
    return default


def resolve_error_reset_flags(args):
    """Parse error-reset mode from args / yaml / --opts."""
    mode = _get_arg(args, 'error_reset_mode', None)
    if mode is not None:
        mode = str(mode).upper()
    if mode in {'NONE', 'OFF', '0', '', 'FALSE'}:
        return False, False
    if mode in {'QUERY', 'SNAPSHOT', 'Q'}:
        return True, False
    if mode in {'SUPPORT', 'SKIP', 'S'}:
        return False, True
    if mode in {'BOTH', 'ALL', 'QS', 'FULL'}:
        return True, True

    use_query = bool(_get_arg(args, 'error_reset_query', False))
    use_support = bool(_get_arg(args, 'error_reset_support', False))
    return use_query, use_support


class MambaErrorReset(nn.Module):
    """Per-layer learnable gates for query snapshot & support skip reset."""

    def __init__(self, num_layers=8, use_query_snapshot=True, use_support_skip=True, gate_init=-2.0):
        super().__init__()
        self.num_layers = num_layers
        self.use_query_snapshot = bool(use_query_snapshot)
        self.use_support_skip = bool(use_support_skip)

        if self.use_query_snapshot:
            self.query_gates = nn.Parameter(torch.full((num_layers,), float(gate_init)))
        else:
            self.query_gates = None

        if self.use_support_skip:
            self.support_gates = nn.Parameter(torch.full((num_layers,), float(gate_init)))
        else:
            self.support_gates = None

    @property
    def enabled(self):
        return self.use_query_snapshot or self.use_support_skip

    def apply_reset(self, q_feat, s_feat, q_init, s_init, layer_idx):
        """Apply reset on spatial tokens [B, H, W, C]."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return q_feat, s_feat

        if self.use_query_snapshot and self.query_gates is not None:
            alpha = torch.sigmoid(self.query_gates[layer_idx])
            q_feat = q_feat + alpha * q_init

        # Mix-Mamba blocks use odd layer_idx (1, 3, 5, 7): re-inject support snapshot.
        if self.use_support_skip and self.support_gates is not None and layer_idx % 2 == 1:
            beta = torch.sigmoid(self.support_gates[layer_idx])
            q_feat = q_feat + beta * s_init

        return q_feat, s_feat


def build_error_reset(args, num_layers=8):
    use_query, use_support = resolve_error_reset_flags(args)
    if not use_query and not use_support:
        return None
    gate_init = float(_get_arg(args, 'error_reset_gate_init', -2.0))
    return MambaErrorReset(
        num_layers=num_layers,
        use_query_snapshot=use_query,
        use_support_skip=use_support,
        gate_init=gate_init,
    )
