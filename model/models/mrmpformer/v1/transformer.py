# -*- coding: utf-8 -*-
"""
MRMPFormer v1 — FDR Transformer。

与 QuanFormer 基线的区别（提示词 §2.1 / §7）：
  QuanFormer 的普通 Transformer Decoder 只把上一层 Query 特征传到下一层；
  本实现在 Decoder 逐层循环中显式增加「边界位置反馈」通路——
  每层解码结束后，通过回调 boundary_pos_fn(k, h_k) 获取基于该层精化
  左右边界生成的位置编码，并按 q_pos^(k+1) = q_pos^base + MLP_pos([xL,xR])
  覆盖下一层的 query position（每次从 base 重新叠加，不做跨层累加）。

Encoder 与 DecoderLayer 结构与 QuanFormer 完全一致（直接复用其实现类），
因此旧 checkpoint 的 encoder/decoder.layers.0 权重可直接加载。

张量布局：
  模块内部沿用 DETR 的 [序列, 批, 维]（[Q, B, D]）布局；
  对外输出统一转为 [K, B, Q, D]（K = 解码层数）。
"""
from typing import Callable, Optional

import torch
from torch import nn, Tensor

# 复用 QuanFormer 的标准 DETR 层实现，保证结构与旧 checkpoint 逐键兼容
from ...quanformer.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    _get_clones,
)


class FDRTransformerDecoder(nn.Module):
    """带边界位置反馈的 Transformer Decoder。

    Args:
        decoder_layer: 标准 DETR 解码层
        num_layers:    解码层数 K（v1 默认 3）
        norm:          输出 LayerNorm（每层中间输出均做 norm，同 return_intermediate）
        boundary_pos_fn: 可选回调 boundary_pos_fn(k, h_k) -> bpos [B, Q, D]。
            在第 k 层解码完成后调用（k < K-1），返回值加到第 k+1 层的
            base query_pos 上。梯度默认全程保留；是否 detach 由模型侧
            配置（detach_boundary_feedback）在回调内部决定。
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos_base: Optional[Tensor] = None,
                boundary_pos_fn: Optional[Callable] = None):
        """
        tgt/memory/query_pos_base: [Q, B, D]；pos: [HW, B, D]
        Returns: hs [K, B, Q, D]（每层 norm 后的中间输出）
        """
        output = tgt
        base_query_pos = query_pos_base
        cur_query_pos = query_pos_base
        intermediate = []

        for k, layer in enumerate(self.layers):
            output = layer(output, memory,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=cur_query_pos)
            normed = self.norm(output) if self.norm is not None else output
            intermediate.append(normed)

            # —— 边界位置反馈：仅在还有下一层时计算 ——
            if boundary_pos_fn is not None and k < self.num_layers - 1:
                # [Q,B,D] → [B,Q,D] 交给模型侧回调（边界解码在 BQD 布局下更直观）
                h_bqd = normed.permute(1, 0, 2)
                bpos = boundary_pos_fn(k, h_bqd)
                if bpos is not None:
                    # q_pos^(k+1) = q_pos^base + bpos（每层从 base 重新叠加）
                    cur_query_pos = base_query_pos + bpos.permute(1, 0, 2)

        # [K,Q,B,D] → [K,B,Q,D]
        return torch.stack(intermediate).permute(0, 2, 1, 3)


class FDRTransformer(nn.Module):
    """Encoder（与 QuanFormer 相同）+ FDRDecoder（逐层边界反馈）。"""

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=1,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = FDRTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        # 与 QuanFormer 相同的初始化策略（Xavier），保证旧权重迁移语义一致
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed,
                boundary_pos_fn: Optional[Callable] = None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [Q, B, D] 作为 base query_pos
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos_base=query_embed,
                          boundary_pos_fn=boundary_pos_fn)
        # hs: [K, B, Q, D]
        return hs, memory.permute(1, 2, 0).view(bs, c, h, w)


def build_fdr_transformer(args):
    return FDRTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
    )
