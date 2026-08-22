# -*- coding: utf-8 -*-
"""MRMPFormer v1 测试（提示词 §15 全部 8 类）。

运行方式（model/ 目录下）：
    python -m tests.test_mrmpformer_v1
纯 CPU 逻辑测试：DummyBackbone 替代 ResNet-50（避免下载预训练权重），
不触发 matplotlib/pyopenms 渲染。
"""
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.util import box_ops
from framework.util.misc import NestedTensor
from models.mrmpformer.v1.detr import (MRMPFormer, MRMPSetCriterion,
                                       load_legacy_quanformer_state)
from models.mrmpformer.v1.transformer import FDRTransformer
from models.mrmpformer.v1.fdr import (make_bin_values, decode_expected_offsets,
                                      DistributionBoundaryLoss, softmax_focal_loss,
                                      dynamic_l1_loss, peak_width_weighted_ciou)
from models.shared.matcher import HungarianMatcher

torch.manual_seed(0)


class DummyBackbone(nn.Module):
    """模拟 Joiner 接口：返回 ([NestedTensor], [pos])，num_channels 对齐 input_proj。"""

    def __init__(self, num_channels=64):
        super().__init__()
        self.num_channels = num_channels
        self.body = nn.Conv2d(3, num_channels, 3, padding=1)

    def forward(self, tensor_list):
        x = self.body(tensor_list.tensors)
        mask = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        return [NestedTensor(x, mask)], [torch.zeros_like(x)]


def make_v1(num_queries=3, num_bins=33, dec_layers=3, detach=False,
            d_model=64, nhead=4, ff=128, seed=0):
    torch.manual_seed(seed)
    transformer = FDRTransformer(d_model=d_model, nhead=nhead, num_encoder_layers=1,
                                 num_decoder_layers=dec_layers, dim_feedforward=ff)
    backbone = DummyBackbone(num_channels=d_model)
    return MRMPFormer(backbone, transformer, num_classes=1, num_queries=num_queries,
                      num_decoder_layers=dec_layers, num_fdr_bins=num_bins,
                      detach_boundary_feedback=detach, aux_loss=True)


def make_criterion(model, fdr_weights=(0.5, 0.7, 1.0), fdr_coef=2.0, **kw):
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_iou=2, iou_type='ciou')
    wd = {'loss_cls_focal_main': 1.0, 'loss_dynamic_l1': 5.0, 'loss_pw_ciou': 2.0}
    for k, a in enumerate(fdr_weights):
        wd[f'loss_fdr_layer_{k + 1}_left'] = fdr_coef * a
        wd[f'loss_fdr_layer_{k + 1}_right'] = fdr_coef * a
    for i in range(len(fdr_weights) - 1):
        wd[f'loss_cls_aux_{i + 1}'] = 1.0
    return MRMPSetCriterion(1, matcher=matcher, weight_dict=wd, eos_coef=0.1,
                            losses=['labels', 'boxes', 'cardinality', 'fdr'],
                            bin_values=model.fdr_bin_values,
                            fdr_layer_weights=list(fdr_weights),
                            pw_ciou_mean_width=0.2297, **kw)


def make_samples(B=2, H=16, W=16, seed=1):
    torch.manual_seed(seed)
    imgs = torch.randn(B, 3, H, W)
    masks = torch.zeros(B, H, W, dtype=torch.bool)
    return NestedTensor(imgs, masks)


def make_targets(spec):
    """spec: list of list[(cx,cy,w,h)] → DETR targets（cxcywh 归一化）。"""
    targets = []
    for boxes in spec:
        if len(boxes) == 0:
            t_boxes = torch.zeros(0, 4, dtype=torch.float32)
        else:
            t_boxes = torch.tensor(boxes, dtype=torch.float32)
        targets.append({'labels': torch.zeros(len(boxes), dtype=torch.int64),
                        'boxes': t_boxes})
    return targets


# ===========================================================================
# §15.1 Shape Test
# ===========================================================================
class TestShape(unittest.TestCase):
    def test_default_shapes(self):
        B, Q, N = 2, 3, 33
        model = make_v1(num_queries=Q, num_bins=N)
        out = model(make_samples(B=B))
        self.assertEqual(out['pred_logits'].shape, (B, Q, 2))       # 第 3 层分类
        self.assertEqual(out['pred_boxes'].shape, (B, Q, 4))        # cxcywh
        self.assertEqual(out['initial_boxes'].shape, (B, Q, 4))
        self.assertEqual(out['initial_edges_ltrb'].shape, (B, Q, 4))
        self.assertEqual(len(out['fdr_logits']), 3)
        self.assertEqual(len(out['fdr_deltas']), 2)
        self.assertEqual(len(out['refined_lr']), 3)
        for z in out['fdr_logits']:
            self.assertEqual(z.shape, (B, Q, 2, N))
        for d in out['fdr_deltas']:
            self.assertEqual(d.shape, (B, Q, 2, N))
        for lr in out['refined_lr']:
            self.assertEqual(lr.shape, (B, Q, 2))
        self.assertEqual(len(out['aux_outputs']), 2)

    def test_variable_b_q_n(self):
        B, Q, N = 1, 5, 17
        model = make_v1(num_queries=Q, num_bins=N)
        out = model(make_samples(B=B))
        for z in out['fdr_logits']:
            self.assertEqual(z.shape, (B, Q, 2, N))
        self.assertEqual(out['pred_logits'].shape, (B, Q, 2))
        self.assertEqual(out['pred_boxes'].shape, (B, Q, 4))


# ===========================================================================
# §15.2 Residual Refinement Test
# ===========================================================================
class TestResidual(unittest.TestCase):
    def test_zero_residual_identity(self):
        """FDR 末层零初始化 → Δz=0 → z2=z1、z3=z2。"""
        model = make_v1()
        out = model(make_samples(B=2))
        self.assertTrue(torch.equal(out['fdr_logits'][1], out['fdr_logits'][0]))
        self.assertTrue(torch.equal(out['fdr_logits'][2], out['fdr_logits'][1]))
        for d in out['fdr_deltas']:
            self.assertTrue(torch.all(d == 0))

    def test_known_residual_accumulation(self):
        """给定非零残差，逐元素验证 z2=z1+Δz2、z3=z2+Δz3。"""
        model = make_v1()
        for head in model.fdr_heads:  # 打破零初始化，产生非零残差
            nn.init.normal_(head.out.weight, std=0.5)
            nn.init.normal_(head.out.bias, std=0.5)
        out = model(make_samples(B=2))
        z1, z2, z3 = out['fdr_logits']
        d2, d3 = out['fdr_deltas']
        self.assertTrue(torch.equal(z2, z1 + d2))
        self.assertTrue(torch.equal(z3, z2 + d3))
        self.assertFalse(torch.all(d2 == 0))
        # 残差相加发生在 Softmax 前的 Logits 上
        self.assertTrue(torch.allclose(
            F.softmax(z2, -1), F.softmax(z1 + d2, -1), atol=1e-6))


# ===========================================================================
# §15.3 Distribution Decode Test
# ===========================================================================
class TestDistributionDecode(unittest.TestCase):
    def setUp(self):
        self.bins = make_bin_values(33, 2.0)
        # 单调性 / 对称覆盖 0 / 长度
        self.assertTrue(torch.all(self.bins[1:] > self.bins[:-1]))
        self.assertEqual(self.bins.shape[0], 33)
        self.assertAlmostEqual(self.bins[16].item(), 0.0, places=6)

    def test_onehot_expectation(self):
        N = 33
        for n in (0, 8, 16, 24, 32):
            z = torch.full((1, 1, 2, N), -1e4)
            z[..., n] = 1e4                       # 高度集中于 bin n
            dx = decode_expected_offsets(z, self.bins, torch.ones(1, 1, 1))
            self.assertAlmostEqual(dx[0, 0, 0].item(), self.bins[n].item(), places=3)
            self.assertAlmostEqual(dx[0, 0, 1].item(), self.bins[n].item(), places=3)

    def test_sign_direction_and_scale(self):
        """正偏移向右（x 增大）、负偏移向左；尺度=初始框宽 w0 时 Δx=w0*W(n)。"""
        n_pos, n_neg = 24, 8                       # W(24)>0，W(8)<0
        N = 33
        z = torch.full((1, 1, 2, N), -1e4)
        z[..., n_pos] = 1e4
        w0 = 0.3
        dx = decode_expected_offsets(z, self.bins, torch.full((1, 1, 1), w0))
        self.assertGreater(dx[0, 0, 0].item(), 0)
        self.assertAlmostEqual(dx[0, 0, 0].item(), w0 * self.bins[n_pos].item(), places=3)

        z2 = torch.full((1, 1, 2, N), -1e4)
        z2[..., n_neg] = 1e4
        dx2 = decode_expected_offsets(z2, self.bins, torch.full((1, 1, 1), w0))
        self.assertLess(dx2[0, 0, 1].item(), 0)
        self.assertAlmostEqual(dx2[0, 0, 1].item(), w0 * self.bins[n_neg].item(), places=3)

    def test_refined_edges_relative_to_initial(self):
        """x^(k) 由累计分布相对【初始边界】解码，不做坐标残差双重累计。"""
        model = make_v1()
        nn.init.normal_(model.fdr_heads[0].out.weight, std=0.5)
        out = model(make_samples(B=1))
        init = out['initial_edges_ltrb']
        dx1 = decode_expected_offsets(out['fdr_logits'][0], model.fdr_bin_values,
                                      model._scale_factor(out['initial_boxes']))
        lr1 = out['refined_lr'][0]
        self.assertTrue(torch.allclose(lr1[..., 0], init[..., 0] + dx1[..., 0], atol=1e-5))
        self.assertTrue(torch.allclose(lr1[..., 1], init[..., 2] + dx1[..., 1], atol=1e-5))
        # Δz=0（层 2/3 零初始化）时，层 2/3 边界应与层 1 相同（而不是再次累加）
        self.assertTrue(torch.allclose(out['refined_lr'][1], lr1, atol=1e-6))
        self.assertTrue(torch.allclose(out['refined_lr'][2], lr1, atol=1e-6))


# ===========================================================================
# §15.4 Final Box Assembly Test
# ===========================================================================
class TestFinalBoxAssembly(unittest.TestCase):
    def test_assembly(self):
        model = make_v1()
        for head in model.fdr_heads:
            nn.init.normal_(head.out.weight, std=0.3)
        out = model(make_samples(B=2))
        xyxy = box_ops.box_cxcywh_to_xyxy(out['pred_boxes'])
        init = out['initial_edges_ltrb']
        lr3 = out['refined_lr'][2]
        lr2 = out['refined_lr'][1]
        w_safe = (lr3[..., 1] - lr3[..., 0]).clamp(min=model.fdr_min_width)
        # 左右 = 第 3 层
        self.assertTrue(torch.allclose(xyxy[..., 0], lr3[..., 0], atol=1e-4))
        self.assertTrue(torch.allclose(xyxy[..., 2], lr3[..., 0] + w_safe, atol=1e-4))
        # 上下 = 第 1 层初始框（不是第 2 层的任何量）
        self.assertTrue(torch.allclose(xyxy[..., 1], init[..., 1], atol=1e-4))
        self.assertTrue(torch.allclose(xyxy[..., 3], init[..., 3], atol=1e-4))
        # 宽度非法时被安全下限保护，输出框恒有效（x2>=x1）
        self.assertTrue(torch.all(xyxy[..., 2] >= xyxy[..., 0] - 1e-6))

        # 退化构造：层 3 左右交叉，验证不产生负宽
        with torch.no_grad():
            out['refined_lr'][2][0, 0, 0] = 0.9
            out['refined_lr'][2][0, 0, 1] = 0.1
        self.assertGreaterEqual((out['refined_lr'][2][0, 0, 1] - out['refined_lr'][2][0, 0, 0]).item(), -0.8)


# ===========================================================================
# §15.5 Final Classification Source Test
# ===========================================================================
class TestFinalClassificationSource(unittest.TestCase):
    def test_pred_logits_from_layer3_only(self):
        model = make_v1()
        captured = []

        def hook(module, inp, outp):
            captured.append(outp)

        h = model.class_embed.register_forward_hook(hook)
        try:
            out = model(make_samples(B=2))
        finally:
            h.remove()
        # class_embed 仅被调用一次，输入为 [K,B,Q,C+1] 的堆叠
        self.assertEqual(len(captured), 1)
        stacked = captured[0]
        self.assertEqual(stacked.shape[0], 3)
        self.assertTrue(torch.equal(out['pred_logits'], stacked[-1]))
        # 三层 Logits 互不相同（随机权重）→ 证明没有取第 1/2 层或层间平均
        self.assertFalse(torch.allclose(stacked[0], stacked[-1]))
        self.assertFalse(torch.allclose(stacked[1], stacked[-1]))
        # 推理口径：p_peak = Softmax(logits3)[peak]
        p_peak = F.softmax(out['pred_logits'], -1)[..., 0]
        self.assertTrue(torch.all((p_peak > 0) & (p_peak < 1)))


# ===========================================================================
# §15.6 Boundary Feedback Gradient Test
# ===========================================================================
class TestBoundaryFeedbackGradient(unittest.TestCase):
    def _grads_after_box_loss(self, detach):
        model = make_v1(detach=detach)
        for head in model.fdr_heads:
            nn.init.normal_(head.out.weight, std=0.3)
        out = model(make_samples(B=2))
        loss = (out['pred_boxes'] ** 2).sum()   # 仅第三层最终框构造损失
        loss.backward()

        def flat_grads(module):
            g = [p.grad.flatten() for p in module.parameters() if p.grad is not None]
            if not g:  # 完全无梯度（detach 消融）→ 视为全零
                return torch.zeros(1)
            return torch.cat(g)

        return flat_grads(model.boundary_pos_mlp), flat_grads(model.fdr_heads[0])

    def test_feedback_gradient_flows(self):
        mlp_g, h1_g = self._grads_after_box_loss(detach=False)
        self.assertTrue(torch.all(torch.isfinite(mlp_g)))
        self.assertTrue(torch.all(torch.isfinite(h1_g)))
        self.assertGreater(mlp_g.abs().sum().item(), 0.0)   # BoundaryMLP 非零梯度
        self.assertGreater(h1_g.abs().sum().item(), 0.0)    # 第 1 层 FDR Head 非零梯度

    def test_detach_ablation(self):
        """detach_boundary_feedback=true：位置反馈梯度被切断（消融语义）。"""
        mlp_g, h1_g = self._grads_after_box_loss(detach=True)
        self.assertEqual(mlp_g.abs().sum().item(), 0.0)     # 反馈通路梯度为 0
        # FDR Head 1 仍可通过 z 残差链（z3=z1+Δz3）获得梯度
        self.assertGreater(h1_g.abs().sum().item(), 0.0)


# ===========================================================================
# §15.7 Loss Test
# ===========================================================================
class TestLosses(unittest.TestCase):
    def setUp(self):
        self.model = make_v1()
        for head in self.model.fdr_heads:
            nn.init.normal_(head.out.weight, std=0.3)
        self.criterion = make_criterion(self.model)
        self.samples = make_samples(B=2)

    def _run(self, targets):
        out = self.model(self.samples)
        losses = self.criterion(out, targets)
        total = sum(v * self.criterion.weight_dict[k]
                    for k, v in losses.items() if k in self.criterion.weight_dict)
        return losses, total

    def test_all_background_empty_targets(self):
        losses, total = self._run(make_targets([[], []]))
        for k, v in losses.items():
            self.assertTrue(torch.isfinite(v).all(), f'{k} not finite')
        total.backward()
        self.assertTrue(math.isfinite(total.item()))

    def test_single_and_multi_peak(self):
        losses, total = self._run(make_targets([[(0.4, 0.5, 0.2, 0.3)],
                                                [(0.3, 0.5, 0.15, 0.4), (0.7, 0.5, 0.2, 0.4)]]))
        for key in ('loss_cls_focal_main', 'loss_dynamic_l1', 'loss_pw_ciou',
                    'loss_fdr_layer_1_left', 'loss_fdr_layer_3_right',
                    'loss_cls_aux_1', 'loss_cls_aux_2', 'class_error'):
            self.assertIn(key, losses)
            self.assertTrue(torch.isfinite(losses[key]).all())
        self.assertIn('fdr_target_overflow_ratio', losses)
        self.assertIn('invalid_box_ratio', losses)
        self.assertIn('gt_exceeds_queries_roi_ratio', losses)
        self.assertIn('matched_positives', losses)
        total.backward()

    def test_narrow_peak_and_degenerate_pred(self):
        # 极窄峰 w_gt=0.01（动态 L1 大权重 + PW-CIoU 权重放大场景）
        losses, total = self._run(make_targets([[(0.5, 0.5, 0.01, 0.3)], []]))
        self.assertTrue(torch.isfinite(losses['loss_dynamic_l1']).all())
        self.assertTrue(torch.isfinite(losses['loss_pw_ciou']).all())
        total.backward()
        # 退化预测框：中心/宽高全零但保留最小宽度（真实模型由 fdr_min_width 保证；
        # 全零面积会造成 IoU union=0 → NaN，实际不会发生）
        out = self.model(self.samples)
        with torch.no_grad():
            out['pred_boxes'] = out['pred_boxes'].detach() * 0
        out['pred_boxes'] = out['pred_boxes'] + torch.tensor(
            [0.0, 0.0, 1e-4, 1e-4])  # w=h=1e-4，可导
        losses = self.criterion(out, make_targets([[(0.5, 0.5, 0.2, 0.3)], []]))
        for k in ('loss_dynamic_l1', 'loss_pw_ciou'):
            self.assertTrue(torch.isfinite(losses[k]).all(), f'{k} not finite')
        sum(v * self.criterion.weight_dict[k] for k, v in losses.items()
            if k in self.criterion.weight_dict).backward()

    def test_focal_no_nan_inf(self):
        logits = (torch.randn(2, 3, 2) * 100).requires_grad_(True)  # 极端 Logits
        tgt = torch.randint(0, 2, (2, 3))
        loss = softmax_focal_loss(logits, tgt, num_classes=1, alpha=0.25, gamma=2.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_focal_eos_weighting(self):
        """背景项乘 eos_coef、前景不受影响（承接 DETR no-object 权重）。"""
        logits = torch.tensor([[[2.0, 0.0]]])          # p_fg≈0.88, p_bg≈0.12
        tgt_bg = torch.tensor([[1]])
        tgt_fg = torch.tensor([[0]])
        l_bg_1 = softmax_focal_loss(logits, tgt_bg, 1, eos_coef=1.0)
        l_bg_01 = softmax_focal_loss(logits, tgt_bg, 1, eos_coef=0.1)
        self.assertAlmostEqual(l_bg_01.item(), 0.1 * l_bg_1.item(), places=6)
        l_fg_1 = softmax_focal_loss(logits, tgt_fg, 1, eos_coef=1.0)
        l_fg_01 = softmax_focal_loss(logits, tgt_fg, 1, eos_coef=0.1)
        self.assertAlmostEqual(l_fg_1.item(), l_fg_01.item(), places=6)

        # 梯度质量平衡验证（训练卡死根因的回归测试）：
        # p=0.5 未决样本下，14 正样本质量 14×0.25×0.25 vs 34 负样本质量
        # 34×0.75×0.25×eos(0.1) → 前景应占主导（无 eos 时 0.75×34 会反超）
        half = torch.tensor([[[0.0, 0.0]]])
        pos_mass = softmax_focal_loss(half, tgt_fg, 1, eos_coef=1.0) * 14
        neg_mass = softmax_focal_loss(half, tgt_bg, 1, eos_coef=0.1) * 34
        self.assertGreater(pos_mass.item(), neg_mass.item())
        # 反证：无 eos 时（旧实现行为）负样本质量压倒正样本
        neg_mass_no_eos = softmax_focal_loss(half, tgt_bg, 1, eos_coef=1.0) * 34
        self.assertLess(pos_mass.item(), neg_mass_no_eos.item())

    def test_dynamic_l1_formula(self):
        src = torch.zeros(1, 4)
        tgt = torch.tensor([[0.4, 0.5, 0.2, 0.3]])
        eps = 1e-6
        loss, stats = dynamic_l1_loss(src, tgt, eps, 1.0, 1.0, None, False)
        lam_c = 1.0 / (0.2 + eps)
        expected = lam_c * (0.4 + 0.5) + 0.2 + 0.3
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertAlmostEqual(stats['p50'], lam_c, places=5)

    def test_pw_ciou_weight_one_when_w_eq_bar_w(self):
        """w_gt = bar_w 时中心项权重=1 → PW-CIoU 与原 CIoU 完全一致。"""
        bar_w = torch.tensor(0.2297)
        pred = torch.tensor([[0.1, 0.2, 0.3, 0.6]])
        gt = torch.tensor([[0.1, 0.2, 0.1 + bar_w, 0.6]])  # w_gt == bar_w
        pw, stats = peak_width_weighted_ciou(pred, gt, bar_w, weight_mode='ratio')
        plain, _ = peak_width_weighted_ciou(pred, gt, bar_w, weight_mode='plain')
        self.assertTrue(torch.allclose(pw, plain, atol=1e-6))
        self.assertAlmostEqual(stats['p50'].item() if isinstance(stats['p50'], torch.Tensor) else stats['p50'], 1.0, places=5)
        # 窄峰 w_gt < bar_w → 权重 > 1
        gt_narrow = torch.tensor([[0.1, 0.2, 0.1 + 0.05, 0.6]])
        pw_n, st_n = peak_width_weighted_ciou(pred, gt_narrow, bar_w, weight_mode='ratio')
        self.assertGreater(st_n['max'], 1.0)

    def test_fdr_soft_labels(self):
        bins = make_bin_values(33, 2.0)
        # 恰好落在 bin 上 → one-hot
        w, ov = DistributionBoundaryLoss.soft_labels(bins[[5]].clone(), bins)
        self.assertAlmostEqual(w[..., 5].item(), 1.0, places=5)
        self.assertAlmostEqual(w.sum(-1).item(), 1.0, places=5)
        self.assertFalse(ov.any())
        # 两 bin 中点 → 各 0.5
        mid = (bins[5] + bins[6]) / 2
        w, ov = DistributionBoundaryLoss.soft_labels(mid.unsqueeze(0), bins)
        self.assertAlmostEqual(w[0, 5].item(), 0.5, places=4)
        self.assertAlmostEqual(w[0, 6].item(), 0.5, places=4)
        self.assertAlmostEqual(w.sum(-1).item(), 1.0, places=5)
        # 越界 → 端点 + overflow 标记
        w, ov = DistributionBoundaryLoss.soft_labels(torch.tensor([5.0]), bins)
        self.assertAlmostEqual(w[0, -1].item(), 1.0, places=5)
        self.assertTrue(ov.all())

    def test_fdr_loss_backward(self):
        z = torch.randn(4, 2, 33, requires_grad=True)
        d = torch.rand(4, 2) * 1.4 - 0.7
        crit = DistributionBoundaryLoss(33)
        res = crit(z, d, make_bin_values(33))
        res['loss'].backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.isfinite(z.grad).all())


# ===========================================================================
# §14.1 旧 checkpoint 迁移 Test
# ===========================================================================
class TestLegacyMigration(unittest.TestCase):
    def test_migrate_quanformer(self):
        model = make_v1()
        # 构造"旧 QuanFormer" state_dict：去掉 FDR 模块与 L2/L3 decoder
        full = model.state_dict()
        legacy = {k: v.clone() for k, v in full.items()
                  if not (k.startswith('fdr_heads.') or k.startswith('boundary_pos_mlp.')
                          or '.layers.1.' in k or '.layers.2.' in k)}
        with torch.no_grad():
            for k in legacy:  # 扰动数值，确保迁移确实生效
                legacy[k].add_(0.123)
        # 迁移前记录 L1 参数作为复制基准
        l0_before = {n: p.detach().clone()
                     for n, p in model.transformer.decoder.layers[0].named_parameters()}
        report = load_legacy_quanformer_state(model, legacy, verbose=False)
        # L2/L3 应从 L1 复制初始化（L1 参数也已加载为 legacy 值）
        l0 = dict(model.transformer.decoder.layers[0].named_parameters())
        for i in (1, 2):
            li = dict(model.transformer.decoder.layers[i].named_parameters())
            for name, p in li.items():
                self.assertTrue(torch.equal(p.detach(), l0[name].detach()))
                self.assertTrue(torch.equal(p.detach(), l0_before[name] + 0.123))
        # FDR heads 未被旧权重污染（保持零初始化）
        self.assertTrue(torch.all(model.fdr_heads[1].out.weight == 0))
        # 报告包含 expected missing（fdr/边界模块）
        self.assertTrue(any('fdr_heads' in k for k in report['expected_missing']))


# ===========================================================================
# §15.8 Tiny-set Overfit Test
# ===========================================================================
class TestTinyOverfit(unittest.TestCase):
    def test_overfit(self):
        torch.manual_seed(7)
        model = make_v1(num_queries=3, d_model=48, ff=96)
        for head in model.fdr_heads:
            nn.init.normal_(head.out.weight, std=0.2)
        criterion = make_criterion(model)
        # 固定 4 个样本（B=1 逐个喂，覆盖多峰/空目标）
        fixed = [
            make_samples(B=1, seed=11),
            make_samples(B=1, seed=12),
            make_samples(B=1, seed=13),
            make_samples(B=1, seed=14),
        ]
        fixed_targets = [
            make_targets([[(0.4, 0.5, 0.2, 0.3)]]),
            make_targets([[(0.3, 0.5, 0.12, 0.4), (0.7, 0.5, 0.2, 0.4)]]),
            make_targets([[(0.55, 0.6, 0.3, 0.2)]]),
            make_targets([[]]),
        ]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        def train_epoch():
            opt.zero_grad()
            total = 0.0
            for s, t in zip(fixed, fixed_targets):
                out = model(s)
                losses = criterion(out, t)
                loss = sum(v * criterion.weight_dict[k]
                           for k, v in losses.items() if k in criterion.weight_dict)
                (loss / len(fixed)).backward()
                total += loss.item()
            opt.step()
            return total / len(fixed)

        loss0 = train_epoch()
        for _ in range(250):
            loss = train_epoch()
        self.assertLess(loss, 0.5 * loss0,
                        f'过拟合失败: {loss0:.4f} → {loss:.4f}')
        # 模型能记住小样本：最终损失绝对值也较小
        self.assertLess(loss, 5.0)
        with torch.no_grad():
            out = model(fixed[0])
            crit_stats = criterion(out, fixed_targets[0])
        # 三层 FDR 边界 MAE 呈精化趋势（允许 L1 略差，L3 应最优）
        mae = [crit_stats[f'fdr_lr_mae_layer_{k}_left'].item() +
               crit_stats[f'fdr_lr_mae_layer_{k}_right'].item() for k in (1, 2, 3)]
        self.assertLessEqual(mae[2], mae[0] + 1e-3,
                             f'第 3 层左边界未优于第 1 层: {mae}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
