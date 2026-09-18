import unittest

import numpy as np

from inference.massnova import _dedup_overlapping_peaks


def _peak(apex_idx, rt_peak, rt_min, rt_max, height, source="signal"):
    return {
        "apex_idx": apex_idx,
        "rt_peak": rt_peak,
        "rt_min": rt_min,
        "rt_max": rt_max,
        "apex_intensity": height,
        "boundary_source": source,
    }


class MassNovaPeakDedupTests(unittest.TestCase):
    def test_small_boundary_overlap_does_not_merge_distinct_peaks(self):
        rt = np.linspace(0.0, 1.0, 11)
        y = np.array([0, 2, 100, 30, 5, 20, 95, 25, 2, 1, 0], dtype=float)
        peaks = [
            _peak(2, 0.2, 0.10, 0.43, 100, "model"),
            _peak(6, 0.6, 0.40, 0.72, 95),
        ]
        result = _dedup_overlapping_peaks(peaks, rt=rt, intensity=y)
        self.assertEqual(len(result), 2)

    def test_deep_valley_prevents_merge_even_with_large_overlap(self):
        rt = np.linspace(0.0, 1.0, 11)
        y = np.array([0, 2, 100, 25, 4, 20, 95, 25, 2, 1, 0], dtype=float)
        peaks = [
            _peak(2, 0.2, 0.05, 0.65, 100, "model"),
            _peak(6, 0.6, 0.30, 0.85, 95),
        ]
        result = _dedup_overlapping_peaks(peaks, apex_tol=0.5, rt=rt, intensity=y)
        self.assertEqual(len(result), 2)

    def test_large_overlap_and_shallow_valley_merges_duplicate_boxes(self):
        rt = np.linspace(0.0, 1.0, 11)
        y = np.array([0, 5, 80, 96, 90, 100, 92, 30, 5, 1, 0], dtype=float)
        peaks = [
            _peak(3, 0.3, 0.10, 0.68, 96, "signal"),
            _peak(5, 0.5, 0.22, 0.75, 100, "model"),
        ]
        result = _dedup_overlapping_peaks(peaks, apex_tol=0.3, rt=rt, intensity=y)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["boundary_source"], "model")


if __name__ == "__main__":
    unittest.main()
