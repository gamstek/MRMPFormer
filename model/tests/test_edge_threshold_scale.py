import unittest

import numpy as np

from inference.massnova import _boundary_stop_levels
from inference.two_round_detection import adjust_first_round_interval


def _broad_peak(rt, apex=5.0, height=10000.0, sigma=2.0, floor=100.0):
    """宽峰 + 常量地板：保证尾区阈值落在峰尾强度范围内，便于观察外推长度。"""
    return floor + height * np.exp(-((rt - apex) ** 2) / (2.0 * sigma ** 2))


def _refine(rt, y, scale):
    return adjust_first_round_interval(
        rt, y, 4.7, 5.3, 3.0, 7.0,
        edge_noise_stop_mode="stable_tail_mean",
        edge_max_span_min=1.0,
        edge_threshold_scale=scale,
        boundary_posterior_lookahead=0,
    )


class EdgeThresholdScaleTests(unittest.TestCase):
    def setUp(self):
        self.rt = np.arange(0.0, 10.0, 0.01)
        self.y = _broad_peak(self.rt)

    def test_default_scale_matches_explicit_one(self):
        """不传该参数（pipeline 兼容路径）必须与显式 1.0 完全一致。"""
        default = adjust_first_round_interval(
            self.rt, self.y, 4.7, 5.3, 3.0, 7.0,
            edge_noise_stop_mode="stable_tail_mean",
            edge_max_span_min=1.0,
            boundary_posterior_lookahead=0,
        )
        self.assertEqual(default, _refine(self.rt, self.y, 1.0))

    def test_lower_scale_extends_right_boundary(self):
        """scale<1 下调阈值 → 需外推更远才截停 → 右边界更靠外、框更宽。"""
        lo1, hi1 = _refine(self.rt, self.y, 1.0)
        lo08, hi08 = _refine(self.rt, self.y, 0.8)
        self.assertGreater(hi08, hi1 + 0.3)
        self.assertGreater((hi08 - lo08), (hi1 - lo1) + 0.3)
        # 左边界对称外扩（同一阈值机制）
        self.assertLess(lo08, lo1)

    def test_half_scale_extends_further_than_point_eight(self):
        """缩放越强，外推越远（单调性）。"""
        _, hi08 = _refine(self.rt, self.y, 0.8)
        _, hi05 = _refine(self.rt, self.y, 0.5)
        self.assertGreater(hi05, hi08)

    def test_invalid_scale_falls_back_to_one(self):
        """0/负值/NaN 等非法缩放回退 1.0，不得把阈值压成 0 导致走到 ROI 尽头。"""
        expected = _refine(self.rt, self.y, 1.0)
        for bad in (0.0, -1.0, float("nan")):
            self.assertEqual(_refine(self.rt, self.y, bad), expected)


class MassNovaStopLevelScaleTests(unittest.TestCase):
    def setUp(self):
        self.rt = np.arange(0.0, 10.0, 0.01)
        self.y = _broad_peak(self.rt)
        self.apex_idx = int(np.argmin(np.abs(self.rt - 5.0)))

    def test_stop_levels_scale_linearly(self):
        left1, right1 = _boundary_stop_levels(
            self.rt, self.y, self.apex_idx, "stable_tail_mean", 1.0)
        left08, right08 = _boundary_stop_levels(
            self.rt, self.y, self.apex_idx, "stable_tail_mean", 1.0,
            edge_threshold_scale=0.8)
        self.assertAlmostEqual(left08, left1 * 0.8, places=6)
        self.assertAlmostEqual(right08, right1 * 0.8, places=6)

    def test_invalid_scale_keeps_original_levels(self):
        left1, right1 = _boundary_stop_levels(
            self.rt, self.y, self.apex_idx, "stable_tail_mean", 1.0)
        left0, right0 = _boundary_stop_levels(
            self.rt, self.y, self.apex_idx, "stable_tail_mean", 1.0,
            edge_threshold_scale=0.0)
        self.assertAlmostEqual(left0, left1, places=9)
        self.assertAlmostEqual(right0, right1, places=9)


if __name__ == "__main__":
    unittest.main()
