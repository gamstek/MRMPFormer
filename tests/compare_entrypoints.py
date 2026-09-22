"""Compare a saved real array case against the MassNova mzML entry point."""
import argparse
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    from utils.mzml_load import load_ms_experiment
    from inference import massnova
    from inference.massnova_runtime import MassNovaArrayRuntime
    from run_parity import compare

    case = json.loads(args.case.read_text(encoding='utf-8'))
    source = case['source']
    manifest = json.loads((args.source / 'manifest.json').read_text(encoding='utf-8'))
    dataset = next(s for s in manifest['golden_sets'] if s['sample'] == source['sample'])
    chrom_index = dataset['chrom_indices'][source['item_index']]
    output = ROOT / 'tests/results/entrypoints' / case['input']['uid']
    output.mkdir(parents=True, exist_ok=True)
    original = args.source / 'source_mzml' / (source['sample'] + '.mzML')
    experiment = load_ms_experiment(str(original), verbose=False)
    chrom = experiment.getChromatograms()[chrom_index]
    rt = np.asarray([point.getRT() for point in chrom])
    intensity = np.asarray([point.getIntensity() for point in chrom])
    np.testing.assert_array_equal(massnova._rt_to_minutes(rt), case['input']['x'])
    intensity_delta = float(np.max(np.abs(np.asarray(intensity, dtype=np.float64) - case['input']['y'])))
    mzml = output / 'single_channel.mzML'
    # Preserve original encoded arrays: reserializing through OpenMS can round
    # intensities to float32 even though the entry reads double-valued points.
    ns = '{http://psi.hupo.org/ms/mzml}'
    document = ET.parse(original).getroot()
    root = document if document.tag == ns + 'mzML' else document.find(ns + 'mzML')
    chromatograms = root.find('.//' + ns + 'chromatogramList')
    selected = list(chromatograms)[chrom_index]
    for child in list(chromatograms):
        if child is not selected:
            chromatograms.remove(child)
    chromatograms.set('count', '1')
    selected.set('index', '0')
    ET.register_namespace('', ns[1:-1])
    ET.ElementTree(root).write(mzml, encoding='utf-8', xml_declaration=True)

    config = case['config']
    model = ROOT / 'build/windows/mrmpformerv2.onnx'
    runtime = MassNovaArrayRuntime(model, config)
    array_result = runtime.process_items([case['input']])['items'][0]
    aligned_input = dict(case['input'], y=[float(point.getIntensity()) for point in chrom])
    aligned_result = runtime.process_items([aligned_input])['items'][0]
    cli = massnova.build_parser().parse_args([])
    for key, value in config.items():
        setattr(cli, key, value)
    cli.model = str(model)
    cli.pipeline_min_chrom_points = config['min_chrom_points']
    cli.pipeline_min_max_intensity = config['min_max_intensity']
    cli.no_plots = True
    captured = []
    original_finalize = massnova.finalize_channel_peaks

    def capture(*args, **kwargs):
        peaks = original_finalize(*args, **kwargs)
        captured.extend(dict(a=p['rt_min'], b=p['rt_max'], c=p['peak_score']) for p in peaks)
        return peaks

    # Observe final values without changing the mzML entry's computation or output.
    with patch.object(massnova, 'finalize_channel_peaks', side_effect=capture):
        file_result = massnova.run_massnova_on_mzml(str(mzml), 'single_channel', cli, output)
    rows = [{k: None if isinstance(v, float) and not math.isfinite(v) else v
             for k, v in row.items()} for row in file_result['peak_rows']]
    def comparison(expected):
        try:
            return dict(passed=True, max_absolute_delta=compare(captured, expected))
        except AssertionError as error:
            return dict(passed=False, error=str(error))

    report = dict(uid=case['input']['uid'], source_mzml=str(original),
                  source_chrom_index=chrom_index, config=config,
                  input_intensity_max_delta=intensity_delta, array_result=array_result,
                  aligned_array_result=aligned_result,
                  mzml_full_precision_peaks=captured, mzml_output_rows=rows,
                  original_input_comparison=comparison(array_result['peaks']),
                  aligned_input_comparison=comparison(aligned_result['peaks']))
    target = output / 'comparison.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f'Report: {target}')
    return 0 if report['original_input_comparison']['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
