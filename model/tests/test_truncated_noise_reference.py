import unittest

import numpy as np

from utils.xic_peak_utils import compute_local_snr, compute_snr_outside_box


def _trace(rt, baseline=1000.0, noise=30.0, seed=7):
    rng = np.random.default_rng(seed)
    return baseline + rng.normal(0.0, noise, size=rt.size)


def _gaussian(rt, apex, height, sigma):
    return height * np.exp(-((rt - apex) ** 2) / (2.0 * sigma ** 2))


class ComputeLocalSnrTruncationTests(unittest.TestCase):
    def test_run_end_truncated_peak_uses_healthy_side_only(self):
        """峰尾被运行末端截断时，右侧扇区不得抬高 p-p 噪声压低 SNR。"""
        rt = np.arange(0.0, 10.0, 0.02)
        y = _trace(rt)
        y += _gaussian(rt, apex=9.85, height=50000.0, sigma=0.08)
        # 峰框右边界 9.95，其后仅剩陡峭下降尾（约 16000→10000）
        snr_fixed = compute_local_snr(rt, y, 9.70, 9.95)
        snr_legacy = compute_local_snr(rt, y, 9.70, 9.95, tail_reject_scale=1e9)
        self.assertTrue(np.isfinite(snr_fixed))
        self.assertGreater(snr_fixed, 100.0)
        self.assertLess(snr_legacy, 30.0)

    def test_both_sides_truncated_falls_back_to_global_quiet(self):
        """双侧扇区均处峰尾（框几乎占满全迹）时回退全迹安静点，不返回 nan。"""
        rt = np.arange(0.0, 10.0, 0.02)
        y = _trace(rt, baseline=100.0, noise=5.0)
        # 宽台地状信号：框外两侧各仅 2-3 点高电平尾
        y[(rt >= 0.06) & (rt <= 9.94)] += 800.0
        snr = compute_local_snr(rt, y, 0.06, 9.94)
        self.assertTrue(np.isfinite(snr))
        self.assertGreater(snr, 20.0)

    def test_interior_peak_keeps_finite_snr(self):
        rt = np.arange(0.0, 10.0, 0.02)
        y = _trace(rt)
        y += _gaussian(rt, apex=5.0, height=20000.0, sigma=0.10)
        snr = compute_local_snr(rt, y, 4.7, 5.3)
        self.assertTrue(np.isfinite(snr))
        self.assertGreater(snr, 20.0)


class ComputeSnrOutsideBoxTruncationTests(unittest.TestCase):
    def test_full_span_box_uses_global_fallback(self):
        """框覆盖整条迹（双侧框外为空）时不再用框内尾巴当噪声。"""
        rt = np.arange(0.0, 10.0, 0.02)
        y = _trace(rt)
        y += _gaussian(rt, apex=5.0, height=20000.0, sigma=0.10)
        snr = compute_snr_outside_box(rt, y, 0.0, 10.0)
        self.assertTrue(np.isfinite(snr))
        self.assertGreater(snr, 20.0)

    def test_steep_right_tail_side_is_rejected(self):
        """右侧框外仍是陡峭峰尾（截断）时该侧不参与噪声估计。"""
        rt = np.arange(0.0, 6.0, 0.02)
        y = _trace(rt)
        y += _gaussian(rt, apex=5.0, height=50000.0, sigma=0.08)
        # 框右边界 5.12 距峰顶 1.5σ，框外 6 点仍在 16000→900 的陡峭尾上
        snr_fixed = compute_snr_outside_box(rt, y, 4.94, 5.12)
        snr_legacy = compute_snr_outside_box(rt, y, 4.94, 5.12, tail_reject_scale=1e9)
        self.assertTrue(np.isfinite(snr_fixed))
        self.assertGreater(snr_fixed, snr_legacy)
        self.assertGreater(snr_fixed, 50.0)


if __name__ == "__main__":
    unittest.main()
