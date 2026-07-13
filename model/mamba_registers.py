"""
Mamba-Reg style register tokens for HMNet Mix-Mamba (support | register | query).

Registers absorb scan artifacts / background noise; outputs on register slots are discarded.
Adapted from Mamba-Reg (VisionMamba cls_token), inserted at support-query boundaries.
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


def mix_seg_len(L_S, num_registers):
    """Per-window segment length after [support | register | query] concat."""
    return 2 * L_S + num_registers


def mix_seq_len(H, W, ratio, num_registers):
    """Total sequence length L for one scan direction group (row/col pair)."""
    L_S = (H // ratio) ** 2
    return (ratio ** 2) * mix_seg_len(L_S, num_registers)


def insert_mix_registers(x_s, x_q, reg_tokens):
    """Insert register tokens between support and query windows.

    Args:
        x_s: [B, 2, Wn, L_S, C] support (after adaptive recap).
        x_q: [B, 2, Wn, L_S, C] query windows.
        reg_tokens: [B, 2, Wn, R, C] expanded register tokens.

    Returns:
        [B, 2, Wn, 2*L_S+R, C]
    """
    return torch.cat([x_s, reg_tokens, x_q], dim=-2)


def expand_register_tokens(reg_tokens, B, n_dirs, n_wins, dtype=None, device=None):
    """Broadcast learnable registers to batch / directions / windows."""
    reg = reg_tokens
    if dtype is not None:
        reg = reg.to(dtype=dtype)
    if device is not None:
        reg = reg.to(device=device)
    # [1, R, C] -> [B, n_dirs, n_wins, R, C]
    return reg.unsqueeze(0).unsqueeze(0).expand(B, n_dirs, n_wins, -1, -1)


class MixMambaRegisters(nn.Module):
    """Shared learnable register tokens for Mix-Mamba odd layers."""

    def __init__(self, dim, num_registers=4):
        super().__init__()
        self.num_registers = int(num_registers)
        if self.num_registers > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_registers, dim))
            trunc_normal_(self.reg_tokens, std=0.02)
        else:
            self.reg_tokens = None

    def forward(self, B, n_dirs, n_wins, dtype, device):
        if self.num_registers <= 0 or self.reg_tokens is None:
            return None
        return expand_register_tokens(
            self.reg_tokens, B, n_dirs, n_wins, dtype=dtype, device=device,
        )
