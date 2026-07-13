"""
Register-augmented gated Mamba blocks (Mamba-Reg + adaptive Recap).

Extends ``mamba_blocks_gated.py`` without modifying it. Register tokens are inserted
between support and query in Mix-Mamba odd layers; register outputs are discarded.
"""
import math
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from model.mamba_blocks import Mlp
from model.mamba_blocks_gated import GatedSS2D
from model.mamba_registers import MixMambaRegisters, insert_mix_registers, mix_seg_len, mix_seq_len
from model.recap_gating import apply_adaptive_recap, modulate_recap_gate_for_layer


class RegisteredGatedSS2D(GatedSS2D):
    """GatedSS2D + Mamba-Reg register tokens in Mix-Mamba."""

    def forward(self, q_feat, s_feat, recap_gate=None, mix_registers=None):
        # Use __dict__ to avoid registering shared modules as child submodules (pollutes state_dict under DDP).
        self.__dict__['_recap_gate'] = recap_gate
        self.__dict__['_mix_registers'] = mix_registers
        try:
            return super(GatedSS2D, self).forward(q_feat, s_feat)
        finally:
            self.__dict__.pop('_recap_gate', None)
            self.__dict__.pop('_mix_registers', None)

    def mix_mamba(self, x_q_idt, x_q_dwn, x_s_dwn):
        B, C, H, W = x_q_idt.shape
        K = 4
        ratio = self.ratio
        size = H // ratio
        L_S = size ** 2
        mix_regs = getattr(self, '_mix_registers', None)
        num_reg = mix_regs.num_registers if mix_regs is not None else 0
        seg = mix_seg_len(L_S, num_reg)
        L = mix_seq_len(H, W, ratio, num_reg)
        q_start = L_S + num_reg

        x_q_idt = x_q_idt.permute(0, 2, 3, 1).contiguous()
        x_s_dwn = x_s_dwn.permute(0, 2, 3, 1).contiguous()

        x_q_row_idt, x_q_col_idt = self.window_partition(x_q_idt, ratio=ratio)
        x_s_row_dwn = x_s_dwn.view(B, 1, -1, C)
        x_s_col_dwn = x_s_dwn.permute(0, 2, 1, 3).contiguous().view(B, 1, -1, C)

        x_q_ori_idt = torch.stack([x_q_row_idt, x_q_col_idt], dim=1)
        x_s_ori_dwn = torch.stack([x_s_row_dwn, x_s_col_dwn], dim=1)

        x_q_inv_idt = x_q_ori_idt.view(B, 2, -1, C)
        x_q_inv_idt = torch.flip(x_q_inv_idt, dims=[-2])
        x_q_inv_idt = x_q_inv_idt.view(B, 2, ratio ** 2, size ** 2, C)

        x_s_inv_dwn = x_s_ori_dwn
        x_s_inv_dwn = torch.flip(x_s_inv_dwn, dims=[-2])

        gate = getattr(self, '_recap_gate', None)
        x_s_ori_dwn = apply_adaptive_recap(x_s_ori_dwn, ratio, gate)
        x_s_inv_dwn = apply_adaptive_recap(x_s_inv_dwn, ratio, gate)

        n_wins = ratio ** 2
        if num_reg > 0:
            reg = mix_regs(B, 2, n_wins, x_q_idt.dtype, x_q_idt.device)
            xs_ori = insert_mix_registers(x_s_ori_dwn, x_q_ori_idt, reg)
            xs_inv = insert_mix_registers(x_s_inv_dwn, x_q_inv_idt, reg)
        else:
            xs_ori = torch.cat([x_s_ori_dwn, x_q_ori_idt], dim=-2)
            xs_inv = torch.cat([x_s_inv_dwn, x_q_inv_idt], dim=-2)

        xs_ori = xs_ori.view(B, 2, -1, C)
        xs_inv = xs_inv.view(B, 2, -1, C)

        xs = torch.cat([xs_ori, xs_inv], dim=1)
        xs = xs.permute(0, 1, 3, 2).contiguous()
        assert xs.size(-1) == L, f"L mismatch: got {xs.size(-1)}, expect {L}"

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        with torch.no_grad():
            xs_s = xs[:, :, :L_S].contiguous()
            dts_s = dts[:, :, :L_S].contiguous()
            Bs_s = Bs[:, :, :, :L_S].contiguous()
            Cs_s = Cs[:, :, :, :L_S].contiguous()
            _, h_s = self.selective_scan(
                xs_s, dts_s, As, Bs_s, Cs_s, Ds, z=None,
                delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=True,
            )

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False,
        )
        out_y = out_y.view(B, K, -1, L)

        out_ori_row, out_ori_col = out_y[:, 0], out_y[:, 1]
        out_inv_row, out_inv_col = out_y[:, 2], out_y[:, 3]

        out_s_ori_row = out_ori_row[:, :, :L_S].contiguous()
        out_s_ori_col = out_ori_col[:, :, :L_S].contiguous()

        out_ori_row = out_ori_row.view(B, C, n_wins, seg)
        out_ori_col = out_ori_col.view(B, C, n_wins, seg)
        out_q_ori_row = out_ori_row[:, :, :, q_start:].contiguous()
        out_q_ori_col = out_ori_col[:, :, :, q_start:].contiguous()
        out_q_ori_row = out_q_ori_row.view(B, C, -1)
        out_q_ori_col = out_q_ori_col.view(B, C, -1)

        out_s_inv_row = out_inv_row[:, :, :L_S].contiguous()
        out_s_inv_col = out_inv_col[:, :, :L_S].contiguous()

        out_inv_row = out_inv_row.view(B, C, n_wins, seg)
        out_inv_col = out_inv_col.view(B, C, n_wins, seg)
        out_q_inv_row = out_inv_row[:, :, :, q_start:].contiguous()
        out_q_inv_col = out_inv_col[:, :, :, q_start:].contiguous()
        out_q_inv_row = out_q_inv_row.view(B, C, -1)
        out_q_inv_col = out_q_inv_col.view(B, C, -1)

        out_s_inv_row = torch.flip(out_s_inv_row, dims=[-1])
        out_s_inv_col = torch.flip(out_s_inv_col, dims=[-1])
        out_q_inv_row = torch.flip(out_q_inv_row, dims=[-1])
        out_q_inv_col = torch.flip(out_q_inv_col, dims=[-1])

        out_q_ori_row = out_q_ori_row.permute(0, 2, 1).contiguous()
        out_q_ori_col = out_q_ori_col.permute(0, 2, 1).contiguous()
        out_q_ori_row = out_q_ori_row.view(B, n_wins, size ** 2, C)
        out_q_ori_col = out_q_ori_col.view(B, n_wins, size ** 2, C)
        out_q_ori_row = self.window_reverse(out_q_ori_row, ratio=ratio, H=H, W=W, mode="row")
        out_q_ori_col = self.window_reverse(out_q_ori_col, ratio=ratio, H=H, W=W, mode="col")

        out_q_inv_row = out_q_inv_row.permute(0, 2, 1).contiguous()
        out_q_inv_col = out_q_inv_col.permute(0, 2, 1).contiguous()
        out_q_inv_row = out_q_inv_row.view(B, n_wins, size ** 2, C)
        out_q_inv_col = out_q_inv_col.view(B, n_wins, size ** 2, C)
        out_q_inv_row = self.window_reverse(out_q_inv_row, ratio=ratio, H=H, W=W, mode="row")
        out_q_inv_col = self.window_reverse(out_q_inv_col, ratio=ratio, H=H, W=W, mode="col")

        out_q_ori_row = out_q_ori_row.view(B, -1, C).permute(0, 2, 1).contiguous()
        out_q_ori_col = out_q_ori_col.view(B, -1, C).permute(0, 2, 1).contiguous()
        out_q_inv_row = out_q_inv_row.view(B, -1, C).permute(0, 2, 1).contiguous()
        out_q_inv_col = out_q_inv_col.view(B, -1, C).permute(0, 2, 1).contiguous()

        out_s_ori_col = out_s_ori_col.view(B, C, size, size).permute(0, 1, 3, 2).contiguous().view(B, C, -1)
        out_s_inv_col = out_s_inv_col.view(B, C, size, size).permute(0, 1, 3, 2).contiguous().view(B, C, -1)

        h_s = h_s.view(B, 4, C, -1).mean(1)
        if gate is not None and 'cross_scale' in gate:
            h_s = h_s * gate['cross_scale'].view(B, 1, 1)

        xs = x_q_dwn
        L_cross = size ** 2
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, 1, -1, L_cross), self.x_proj_weight[0:1].contiguous())
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, 1, -1, L_cross), self.dt_projs_weight[0:1].contiguous())

        xs = xs.float().view(B, -1, L_cross)
        dts = dts.contiguous().float().view(B, -1, L_cross)
        Bs = Bs.float().view(B, 1, -1, L_cross)
        Cs = Cs.float().view(B, 1, -1, L_cross)
        Ds = self.Ds.float()[0].contiguous().view(-1)
        As = -torch.exp(self.A_logs.float().view(K, C, -1)[0].contiguous())
        dt_projs_bias = self.dt_projs_bias.float()[0].contiguous().view(-1)

        out_y = self.selective_scan_cross(
            h_s, xs, dts, As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias, delta_softplus=True,
        ).view(B, 1, -1, L_cross)
        out_y_q = out_y[:, 0]

        return (out_q_ori_row, out_q_ori_col, out_q_inv_row, out_q_inv_col, out_y_q), (
            out_s_ori_row, out_s_ori_col, out_s_inv_row, out_s_inv_col)


class RegisteredGatedVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        ratio: int = 4,
        layer_idx: int = 0,
        mlp_ratio: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = RegisteredGatedSS2D(
            d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,
            ratio=ratio, layer_idx=layer_idx, **kwargs,
        )
        self.drop_path = DropPath(drop_path)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp_q = Mlp(in_features=hidden_dim, hidden_features=hidden_dim * mlp_ratio)
        self.mlp_s = Mlp(in_features=hidden_dim, hidden_features=hidden_dim * mlp_ratio)

    def forward(self, q_feat, s_feat, recap_gate=None, mix_registers=None):
        q_skip, s_skip = q_feat, s_feat
        layer_gate = modulate_recap_gate_for_layer(recap_gate, self.layer_idx)
        q_feat, s_feat = self.self_attention(
            self.ln_1(q_feat), self.ln_1(s_feat),
            recap_gate=layer_gate, mix_registers=mix_registers,
        )
        q_feat = q_skip + self.drop_path(q_feat)
        s_feat = s_skip + self.drop_path(s_feat)
        q_feat = q_feat + self.drop_path(self.mlp_q(self.ln_2(q_feat)))
        s_feat = s_feat + self.drop_path(self.mlp_s(self.ln_2(s_feat)))
        return q_feat, s_feat


class RegisteredGatedVSSLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        d_state=16,
        mlp_ratio=1,
        **kwargs,
    ):
        super().__init__()
        self.ratio = 4
        if isinstance(drop_path, list):
            drop_path_list = drop_path
        else:
            drop_path_list = [drop_path] * depth
        self.blocks = nn.ModuleList([
            RegisteredGatedVSSBlock(
                hidden_dim=dim,
                drop_path=drop_path_list[i],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                ratio=self.ratio,
                layer_idx=i,
                mlp_ratio=mlp_ratio,
                **kwargs,
            )
            for i in range(depth)
        ])

    def forward(self, q_feat, s_feat, recap_gate=None, mix_registers=None):
        for blk in self.blocks:
            q_feat, s_feat = blk(q_feat, s_feat, recap_gate=recap_gate, mix_registers=mix_registers)
        return q_feat, s_feat


class RegisteredGatedVSSM(nn.Module):
    """VSSM with MSFM-style recap gating + Mamba-Reg registers in Mix-Mamba."""

    def __init__(
        self,
        depths=(8,),
        dims=(256,),
        mlp_ratio=1,
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        num_registers=4,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.num_registers = int(num_registers)

        expand = kwargs.get('expand', 2)
        inner_dim = int(dims[0] * expand)
        self.mix_registers = MixMambaRegisters(inner_dim, num_registers=self.num_registers)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RegisteredGatedVSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                **kwargs,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q_feat, s_feat, recap_gate: Optional[dict] = None):
        q_feat = rearrange(q_feat, 'b c h w -> b h w c')
        s_feat = rearrange(s_feat, 'b c h w -> b h w c')
        q_feat = self.pos_drop(q_feat)
        s_feat = self.pos_drop(s_feat)

        mix_regs = self.mix_registers if self.num_registers > 0 else None
        for layer in self.layers:
            q_feat, s_feat = layer(q_feat, s_feat, recap_gate=recap_gate, mix_registers=mix_regs)

        q_feat = self.norm(q_feat)
        q_feat = rearrange(q_feat, 'b h w c -> b c h w')
        return q_feat
