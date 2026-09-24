"""Regression checks for the wide-peak dataset builder."""

import unittest

import numpy as np

from tools.build_wide_finetune import (
    add_image,
    contains_all,
    noise_variant,
    parse_number,
    valid_intervals,
    window_for,
)


class WideFinetuneBuilderTests(unittest.TestCase):
    def test_parses_rt_before_parenthesized_intensity(self):
        self.assertEqual(parse_number("3.037(55.000)"), 3.037)

    def test_rejects_half_interval(self):
        intervals, reason = valid_intervals({"peak_start1": "3.0", "peak_end1": None})
        self.assertIsNone(intervals)
        self.assertEqual(reason, "invalid_boundary")

    def test_rounded_acquisition_edge_is_allowed_but_real_crop_is_not(self):
        rt = np.linspace(0.0, 5.0, 501)
        window = window_for(4.0, rt)
        self.assertTrue(contains_all([(3.2, 5.0005)], window))
        self.assertFalse(contains_all([(3.2, 5.02)], window))

    def test_box_tracks_actual_window_and_clamps_rounding(self):
        dataset = {"images": [], "annotations": []}
        add_image(dataset, "example.jpeg", [(3.2, 5.0005)], (3.0, 5.0), {})
        self.assertEqual(dataset["annotations"][0]["bbox"], [40.0, 0.0, 360.0, 300.0])

    def test_noise_never_makes_intensity_negative(self):
        data = np.array([0.0, 0.0, 0.2, 5.0, 0.2, 0.0])
        varied = noise_variant(data, np.random.default_rng(42))
        self.assertEqual(varied.shape, data.shape)
        self.assertGreaterEqual(float(varied.min()), 0.0)


if __name__ == "__main__":
    unittest.main()
