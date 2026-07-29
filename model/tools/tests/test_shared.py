# -*- coding: utf-8 -*-
"""_shared 模块单元测试。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# 确保可以导入 _shared 模块
_COPY_ROOT = Path(__file__).resolve().parent.parent
if str(_COPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_COPY_ROOT))

from _shared.table_io import normalize_compound_name, parse_area
from _shared.chrom_json import parse_q1_q3, parse_time_intensity
from _shared.artifacts import image_to_row_index, resolve_rt_window, safe_float


class TestNormalizeCompoundName(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(normalize_compound_name(" 阿维菌素 "), "阿维菌素")

    def test_casefold(self):
        self.assertEqual(normalize_compound_name("ABC"), "abc")

    def test_empty(self):
        self.assertEqual(normalize_compound_name(""), "")

    def test_none(self):
        result = normalize_compound_name(None)
        self.assertEqual(result, "none")  # str(None) = "None" → casefold → "none"


class TestParseArea(unittest.TestCase):
    def test_float(self):
        self.assertAlmostEqual(parse_area("123.45"), 123.45)

    def test_int(self):
        self.assertAlmostEqual(parse_area(100), 100.0)

    def test_scientific(self):
        self.assertAlmostEqual(parse_area("1.23e5"), 123000.0)

    def test_scientific_with_space(self):
        self.assertAlmostEqual(parse_area("1.23 e 5"), 123000.0)

    def test_na(self):
        self.assertIsNone(parse_area("N/A"))
        self.assertIsNone(parse_area("NA"))
        self.assertIsNone(parse_area("nan"))

    def test_less_than_2_points(self):
        self.assertIsNone(parse_area("<2 POINTS"))

    def test_empty(self):
        self.assertIsNone(parse_area(""))

    def test_none(self):
        self.assertIsNone(parse_area(None))

    def test_nan(self):
        self.assertIsNone(parse_area(float("nan")))

    def test_thousands_separator(self):
        self.assertAlmostEqual(parse_area("1,234.56"), 1234.56)


class TestParseQ1Q3(unittest.TestCase):
    def test_dict_format(self):
        data = {"mz": {"precursor_mz": 384.2, "product_mz": 247.1}}
        q1, q3 = parse_q1_q3(data)
        self.assertAlmostEqual(q1, 384.2)
        self.assertAlmostEqual(q3, 247.1)

    def test_scalar_format(self):
        data = {"mz": 384.2, "q3": 247.1}
        q1, q3 = parse_q1_q3(data)
        self.assertAlmostEqual(q1, 384.2)
        self.assertAlmostEqual(q3, 247.1)

    def test_missing(self):
        data = {}
        q1, q3 = parse_q1_q3(data)
        self.assertIsNone(q1)
        self.assertIsNone(q3)

    def test_invalid_string(self):
        data = {"mz": "abc", "q3": "def"}
        q1, q3 = parse_q1_q3(data)
        self.assertIsNone(q1)
        self.assertIsNone(q3)


class TestParseTimeIntensity(unittest.TestCase):
    def test_normal_minute(self):
        data = {
            "time": {"values": [0, 1, 2], "unit": "minute"},
            "intensity": {"values": [100, 200, 150]},
        }
        rt, inten = parse_time_intensity(data)
        self.assertIsNotNone(rt)
        self.assertIsNotNone(inten)
        self.assertEqual(len(rt), 3)
        self.assertAlmostEqual(rt[0], 0.0)
        self.assertAlmostEqual(rt[1], 60.0)  # minute → second

    def test_second_unit(self):
        data = {
            "time": {"values": [0, 1, 2], "unit": "second"},
            "intensity": {"values": [100, 200, 150]},
        }
        rt, inten = parse_time_intensity(data)
        self.assertAlmostEqual(rt[1], 1.0)  # already seconds

    def test_too_short(self):
        data = {"time": {"values": [0]}, "intensity": {"values": [100]}}
        rt, inten = parse_time_intensity(data)
        self.assertIsNone(rt)
        self.assertIsNone(inten)

    def test_length_mismatch(self):
        data = {"time": {"values": [0, 1]}, "intensity": {"values": [100]}}
        rt, inten = parse_time_intensity(data)
        self.assertIsNone(rt)


class TestResolveRtWindow(unittest.TestCase):
    def setUp(self):
        self.roi_map = {
            "image_001.jpeg": (1.0, 2.0),
            "筛选保留/image_002.jpeg": (3.0, 4.0),
        }

    def test_exact_match(self):
        w, note = resolve_rt_window(self.roi_map, "image_001.jpeg")
        self.assertEqual(w, (1.0, 2.0))
        self.assertIn("key=", note)

    def test_basename_match(self):
        w, note = resolve_rt_window(self.roi_map, "some/path/image_001.jpeg")
        self.assertIsNotNone(w)

    def test_no_match(self):
        w, note = resolve_rt_window(self.roi_map, "nonexistent.jpeg")
        self.assertIsNone(w)
        self.assertIn("no_match", note)

    def test_empty(self):
        w, note = resolve_rt_window(self.roi_map, "")
        self.assertIsNone(w)


class TestImageToRowIndex(unittest.TestCase):
    def test_compound_name_numeric(self):
        idx = image_to_row_index("anything.jpeg", "5")
        self.assertEqual(idx, 4)  # 1-based → 0-based = 4

    def test_image_prefix(self):
        idx = image_to_row_index("27_mz384.2000_q3247.1000_snr27.8987_refined.png", "abc")
        self.assertEqual(idx, 26)  # 27 → 26

    def test_no_match(self):
        idx = image_to_row_index("no_number_here.png", "abc")
        self.assertIsNone(idx)

    def test_boundary_zero(self):
        idx = image_to_row_index("1_mz100_q3200.png", None)
        self.assertEqual(idx, 0)

    def test_boundary_invalid(self):
        idx = image_to_row_index("0_mz100_q3200.png", None)
        self.assertIsNone(idx)


class TestSafeFloat(unittest.TestCase):
    def test_normal(self):
        self.assertAlmostEqual(safe_float("3.14"), 3.14)

    def test_nan(self):
        self.assertTrue(np.isnan(safe_float(float("nan"))))

    def test_invalid(self):
        self.assertTrue(np.isnan(safe_float("abc")))

    def test_default(self):
        self.assertEqual(safe_float("abc", -1.0), -1.0)


if __name__ == "__main__":
    unittest.main()
