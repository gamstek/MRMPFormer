import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from inference import massnova

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / 'configs/massnova.json').read_text(encoding='utf-8'))
config.update(no_plots=True, plot=False, keep_windows=False)
args = SimpleNamespace(**config)
extract = massnova.extract_full_xics


def selected(*values, **kwargs):
    features, excluded = extract(*values, **kwargs)
    return [f for f in features if f['chrom_index'] in {388, 641}], []


with patch.object(massnova, 'extract_full_xics', side_effect=selected):
    info = massnova.run_massnova_on_mzml(
        str(ROOT.parent / 'data/mzml/test3/test3_3.mzML'), 'test3_3', args,
        ROOT / 'project_review/release_smoke_20260918')
rows = info['peak_rows']
assert rows and all(np.isfinite(r['peak_score']) and r['area'] > 0 for r in rows)
napropamide = [r for r in rows if r['chrom_index'] == 388]
assert len(napropamide) == 3, napropamide
assert any(abs(r['rt_peak'] - 7.9817) < 0.01 for r in napropamide)
print(json.dumps({'channels': info['n_channels'], 'peaks': info['n_peaks'], 'rows': rows}, ensure_ascii=True, indent=2))
