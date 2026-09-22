# -*- coding: utf-8 -*-
"""临时探针：查看 test3 mzML 头部（spectrum/chromatogram 计数）。用完即删。"""
import re
import sys

path = r"d:\yinlibo\MRMPFormer\data\mzml\test3\test3_1.mzML"
head = open(path, encoding="utf-8", errors="ignore").read(300000)
for label, pat in [
    ("specCount", r'spectrumList count="(\d+)"'),
    ("chromCount", r'chromatogramList count="(\d+)"'),
    ("firstChromID", r'<chromatogram id="([^"]+)"'),
    ("firstSpectrumID", r'<spectrum id="([^"]+)"'),
]:
    m = re.search(pat, head)
    print(label, "=", m.group(1) if m else None)
