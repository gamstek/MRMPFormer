"""Tests for standards-aware mzML chromatogram binary decoding."""

import base64
import os
import tempfile
import unittest
import zlib

import numpy as np

from io_mzml import read_mzml_chromatograms


NS = "http://psi.hupo.org/ms/mzml"


def _binary_array(values, dtype_accession, array_accession, unit_accession=None,
                  compressed=False):
    dtype = {
        "MS:1000521": np.dtype("<f4"),
        "MS:1000523": np.dtype("<f8"),
        "MS:1000519": np.dtype("<i4"),
        "MS:1000522": np.dtype("<i8"),
    }[dtype_accession]
    payload = np.asarray(values, dtype=dtype).tobytes()
    if compressed:
        payload = zlib.compress(payload)
    encoded = base64.b64encode(payload).decode("ascii")
    unit = f' unitAccession="{unit_accession}"' if unit_accession else ""
    compression = "MS:1000574" if compressed else "MS:1000576"
    return f"""
      <binaryDataArray>
        <cvParam accession="{array_accession}"{unit}/>
        <cvParam accession="{dtype_accession}"/>
        <cvParam accession="{compression}"/>
        <binary>{encoded}</binary>
      </binaryDataArray>"""


def _mzml(time_array, intensity_array):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<mzML xmlns="{NS}">
  <run><chromatogramList count="1"><chromatogram id="test">
    <binaryDataArrayList count="2">{time_array}{intensity_array}
    </binaryDataArrayList>
  </chromatogram></chromatogramList></run>
</mzML>"""


class ReadMzmlChromatogramsTests(unittest.TestCase):
    def _read(self, xml):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "test.mzML")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(xml)
            return read_mzml_chromatograms(path)

    def test_decodes_float32_and_float64_without_compression(self):
        xml = _mzml(
            _binary_array([0.0, 0.5, 1.0], "MS:1000521", "MS:1000595", "UO:0000010"),
            _binary_array([10.0, 20.0, 30.0], "MS:1000523", "MS:1000515"),
        )
        chromatogram = self._read(xml)[0]
        np.testing.assert_allclose(chromatogram["rt"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(chromatogram["intensity"], [10.0, 20.0, 30.0])
        self.assertEqual(chromatogram["rt"].dtype, np.dtype("<f4"))
        self.assertEqual(chromatogram["intensity"].dtype, np.dtype("<f8"))

    def test_decodes_zlib_and_converts_minutes_to_seconds(self):
        xml = _mzml(
            _binary_array([0.0, 0.5, 1.0], "MS:1000523", "MS:1000595", "UO:0000031", True),
            _binary_array([1, 2, 3], "MS:1000519", "MS:1000515", compressed=True),
        )
        chromatogram = self._read(xml)[0]
        np.testing.assert_allclose(chromatogram["rt"], [0.0, 30.0, 60.0])
        np.testing.assert_array_equal(chromatogram["intensity"], [1, 2, 3])

    def test_accepts_empty_arrays(self):
        xml = _mzml(
            _binary_array([], "MS:1000523", "MS:1000595", "UO:0000010"),
            _binary_array([], "MS:1000523", "MS:1000515"),
        )
        chromatogram = self._read(xml)[0]
        self.assertEqual(len(chromatogram["rt"]), 0)
        self.assertEqual(len(chromatogram["intensity"]), 0)

    def test_rejects_missing_numeric_type(self):
        time_array = """
          <binaryDataArray>
            <cvParam accession="MS:1000595" unitAccession="UO:0000010"/>
            <cvParam accession="MS:1000576"/>
            <binary></binary>
          </binaryDataArray>"""
        intensity = _binary_array([], "MS:1000523", "MS:1000515")
        with self.assertRaisesRegex(ValueError, "numeric type"):
            self._read(_mzml(time_array, intensity))

    def test_rejects_mismatched_array_lengths(self):
        xml = _mzml(
            _binary_array([0.0, 1.0], "MS:1000523", "MS:1000595", "UO:0000010"),
            _binary_array([10.0], "MS:1000523", "MS:1000515"),
        )
        with self.assertRaisesRegex(ValueError, "different lengths"):
            self._read(xml)


if __name__ == "__main__":
    unittest.main()
