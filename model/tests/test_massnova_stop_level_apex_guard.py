# -*- coding: utf-8 -*-
"""单侧停阈值护栏 + 模型原始框保留 的回归测试。

背景（test2 chrom052 甲羧除草醚-2，RT≈16.0 候选）：
该候选右侧 ~1 min 处存在更高的邻近峰，stable_tail_mean 尾窗（apex±1min）把邻居峰翼
当成本侧基线，估计出的停阈值达到 apex 的 1.05~1.11 倍 → 内审在 apex 后一步就截停，
峰只积分了一半（面积仅为完整峰的一半）。护栏要求停阈值必须显著低于本峰 apex。
"""
import unittest

import numpy as np

from inference.massnova import (
    _boundary_stop_levels,
    finalize_channel_peaks,
    refine_model_boundaries,
)


def _two_peaks(rt, low_apex=16.0, low_h=3.1e6, high_apex=16.97, high_h=4.3e6,
               sigma=0.13, valley=8e3):
    """低峰 + 1 min 外更高邻居峰，谷底 ~valley（复现真实通道形态）。"""
    y = valley + low_h * np.exp(-((rt - low_apex) ** 2) / (2.0 * sigma ** 2)) \
        + high_h * np.exp(-((rt - high_apex) ** 2) / (2.0 * sigma ** 2))
    return y


SCAN_PARAMS = {
    "baseline_percentile": 25.0, "baseline_mode": "global_percentile",
    "min_peak_ratio": 0.05, "prominence_ratio": 0.055, "min_prominence_abs": 0.0,
    "min_peak_gap_points": 3, "min_peak_width_min": 0.10, "void_time_min": 0.5,
    "max_peaks_per_channel": 50, "valley_ratio": 0.90, "min_valley_central_frac": 0.12,
    "init_half_width_min": 0.05, "boundary_posterior_lookahead": 5,
    "boundary_posterior_mean_scale": 1.25, "edge_noise_stop_mode": "stable_tail_mean",
    "edge_max_span_min": 1.0, "edge_threshold_scale": 0.8, "min_snr": 15.0,
    "min_peak_span_points": 5, "min_area": 0.0, "window_half_min": 1.0,
    "dup_apex_tol": 0.2, "dup_min_overlap_fraction": 0.25,
    "dup_shallow_valley_min_ratio": 0.70, "signal_score_snr_pivot": 10.0,
    "signal_score_points_good": 10.0, "signal_score_snr_weight": 0.8,
    "signal_score_points_weight": 0.2, "width_fuse_ratio": 1.5,
    "model_boundary_baseline_ratio": 0.01,
}


class StopLevelApexGuardTests(unittest.TestCase):
    def setUp(self):
        self.rt = np.arange(14.0, 18.5, 0.01)
        self.y = _two_peaks(self.rt)
        self.apex_idx = int(np.argmin(np.abs(self.rt - 16.0)))

    def test_contaminated_side_is_capped_below_apex(self):
        """邻居峰污染该侧估计时，停阈值必须被压到 apex 的显著比例以下。"""
        apex = float(self.y[self.apex_idx])
        left, right = _boundary_stop_levels(
            self.rt, self.y, self.apex_idx, "stable_tail_mean", 1.0,
            edge_threshold_scale=1.0, max_stop_apex_ratio=1e9)
        self.assertGreater(right, apex)  # 原始估计确实被邻居抬高到 apex 之上
        _, guarded = _boundary_stop_levels(
            self.rt, self.y, self.apex_idx, "stable_tail_mean", 1.0,
            edge_threshold_scale=1.0)
        self.assertLess(guarded, apex * 0.2 + 1e-6)
        self.assertLess(left, apex)  # 未受污染的左侧不受影响

    def test_right_boundary_not_truncated_at_apex(self):
        """护栏生效后右边界必须越过 apex 一段距离，而不是停在 apex 后一步。"""
        model_box = (15.9384, 16.3844)
        cand = {"apex_idx": self.apex_idx, "rt_min": model_box[0], "rt_max": model_box[1]}
        lo, hi = refine_model_boundaries(self.rt, self.y, [cand], SCAN_PARAMS)[0]
        apex_rt = float(self.rt[self.apex_idx])
        self.assertGreater(hi - apex_rt, 0.05)
        # 右侧至少保留模型框该侧跨度的一半
        self.assertGreater(apex_rt - lo, 0.05)

    def test_isolated_peak_level_is_unchanged(self):
        """无高邻居时估计值低于 apex，护栏不得改变原有水平。"""
        rt = np.arange(10.0, 11.6, 0.01)
        y = 100.0 + 5.0e5 * np.exp(-((rt - 10.8) ** 2) / (2.0 * 0.05 ** 2))
        idx = int(np.argmin(np.abs(rt - 10.8)))
        raw = _boundary_stop_levels(rt, y, idx, "stable_tail_mean", 1.0,
                                    edge_threshold_scale=0.8, max_stop_apex_ratio=1e9)
        guarded = _boundary_stop_levels(rt, y, idx, "stable_tail_mean", 1.0,
                                        edge_threshold_scale=0.8)
        self.assertEqual(raw, guarded)


class ModelSeedBoundsTests(unittest.TestCase):
    def test_output_peak_keeps_model_box_before_refinement(self):
        """模型原始框必须保留（model_rt_*），且不等于精修后的最终边界。"""
        rt = np.arange(14.0, 18.5, 0.01)
        y = _two_peaks(rt)
        apex_idx = int(np.argmin(np.abs(rt - 16.0)))
        model_box = (15.90, 16.45)
        candidates = [{
            "apex_idx": apex_idx, "rt_peak": float(rt[apex_idx]),
            "apex_intensity": float(y[apex_idx]), "model_score": 0.95, "validated": True,
            "rt_min": model_box[0], "rt_max": model_box[1],
            "model_rt_min": model_box[0], "model_rt_max": model_box[1],
        }]
        peaks = finalize_channel_peaks(rt, y, candidates, SCAN_PARAMS, threshold=0.6)
        self.assertEqual(len(peaks), 1)
        peak = peaks[0]
        self.assertAlmostEqual(peak["model_rt_min"], model_box[0], places=6)
        self.assertAlmostEqual(peak["model_rt_max"], model_box[1], places=6)
        self.assertNotEqual(round(peak["rt_min"], 4), round(peak["model_rt_min"], 4))


if __name__ == "__main__":
    unittest.main()
