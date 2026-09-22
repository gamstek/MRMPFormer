"""Read saved inputs and compare source Python, compiled Cython, and the C DLL."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
from import_inputs import load_cases


def compare(actual, expected, path='result'):
    """Strict structure/order plus finite numeric comparison; report exact field."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f'{path}: keys differ: {actual!r} vs {expected!r}')
        return max((compare(actual[k], v, f'{path}.{k}') for k, v in expected.items()), default=0.0)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f'{path}: count differs: {actual!r} vs {expected!r}')
        return max((compare(a, b, f'{path}[{i}]') for i, (a, b) in enumerate(zip(actual, expected))), default=0.0)
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not isinstance(actual, (float, int)) or not math.isfinite(actual) or not math.isfinite(expected):
            raise AssertionError(f'{path}: invalid numeric value {actual!r} vs {expected!r}')
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise AssertionError(f'{path}: {actual!r} != {expected!r}')
        return abs(actual - expected)
    if actual != expected:
        raise AssertionError(f'{path}: {actual!r} != {expected!r}')
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT / 'build/windows')
    parser.add_argument('--runner', type=Path, default=ROOT / 'tests/build/Release/inference_parity_runner.exe')
    parser.add_argument('--output', type=Path, default=ROOT / 'tests/results')
    parser.add_argument('--inputs', type=Path, default=ROOT / 'tests/inputs')
    args = parser.parse_args()
    package, runner, output = args.package.resolve(), args.runner.resolve(), args.output.resolve()
    model = package / 'mrmpformerv2.onnx'
    for file in (model, package / 'mrmpformer.dll'):
        if not file.is_file(): parser.error(f'Missing {file}; build the package and C++ tests first.')
    output.mkdir(parents=True, exist_ok=True)
    suite = load_cases(args.inputs)
    subprocess.run(['cmake', '-S', str(ROOT / 'tests'), '-B', str(ROOT / 'tests/build'),
                    '-DPACKAGE_DIR=' + str(package)], check=True)
    subprocess.run(['cmake', '--build', str(ROOT / 'tests/build'), '--config', 'Release'], check=True)
    (output / 'inputs.json').write_text(json.dumps(suite, allow_nan=False), encoding='utf-8')
    # A separate executable loads the shipped DLL and private Python, never this interpreter.
    with tempfile.TemporaryDirectory(prefix='mrmpformer-parity-') as folder:
        work = Path(folder)
        shutil.copy2(runner, work / 'runner.exe')
        shutil.copy2(output / 'inputs.json', work / 'inputs.json')
        env = {k: v for k, v in os.environ.items()
               if not k.upper().startswith(('PYTHON', 'CONDA', 'VIRTUAL_ENV', 'MRMPFORMER'))}
        env['PATH'] = str(package) + os.pathsep + str(Path(os.environ['SystemRoot']) / 'System32')
        env['PYTHONHOME'] = str(work / 'nonexistent')
        env['PYTHONPATH'] = str(work / 'nonexistent')
        env['MPLCONFIGDIR'] = str(work / 'matplotlib')
        print(f'Loaded {sum(len(g["items"]) for g in suite)} files from {args.inputs}; running C APIs...', flush=True)
        result = subprocess.run([str(work / 'runner.exe'), str(model), 'inputs.json', 'actual.json'],
                                cwd=work, env=env, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=900)
        (output / 'c-process.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            print(f'C process log: {output / "c-process.log"}', file=sys.stderr)
        result.check_returncode()
        shutil.copy2(work / 'actual.json', output / 'actual.json')
    actual = json.loads((output / 'actual.json').read_text(encoding='utf-8'))
    subprocess.run([sys.executable, str(ROOT / 'tests/compiled_runner.py'), str(model),
                    str(output / 'inputs.json'), str(output / 'compiled.json')], check=True, timeout=900)
    compiled_payload = json.loads((output / 'compiled.json').read_text(encoding='utf-8'))
    compiled = compiled_payload['groups']
    sys.path.insert(0, str(ROOT / 'model'))
    from inference import massnova_runtime, massnova, onnx_window_predictor
    for module in (massnova_runtime, massnova, onnx_window_predictor):
        if Path(module.__file__).suffix != '.py':
            raise RuntimeError(f'Reference must use source Python: {module.__file__}')
    expected = []
    expected_single = []
    for index, group in enumerate(suite):
        print(f'Running source Python reference {index + 1}/{len(suite)}...', flush=True)
        runtime = massnova_runtime.MassNovaArrayRuntime(model, group['config'])
        expected.append(runtime.process_items(group['items']))
        # ONNX may round float32 scores differently for different batch shapes.
        # Compare each ABI against the same Python invocation granularity.
        expected_single.append([runtime.process_items([item])['items'][0] for item in group['items']])
        del runtime
    (output / 'expected.json').write_text(
        json.dumps({'batch': expected, 'single': expected_single}, indent=2), encoding='utf-8')
    if len(actual) != len(suite) or len(compiled) != len(suite):
        raise AssertionError('C/compiled result group count mismatch')
    records = []
    class ParityTests(unittest.TestCase):
        pass
    for group_index, group in enumerate(suite):
        if len(actual[group_index]['batch']['items']) != len(group['items']) or len(actual[group_index]['single']) != len(group['items']):
            raise AssertionError('C result item count mismatch')
        if len(compiled[group_index]['batch']['items']) != len(group['items']) or len(compiled[group_index]['single']) != len(group['items']):
            raise AssertionError('Compiled result item count mismatch')
        for index, item in enumerate(group['items']):
            def test(self, gi=group_index, i=index, source=item):
                record = dict(uid=source['uid'], passed=False)
                records.append(record)
                reference = expected[gi]['items'][i]
                provenance = json.loads((args.inputs / (source['uid'] + '.json')).read_text(encoding='utf-8')).get('source', {})
                detail = dict(input=source, source=provenance, config=suite[gi]['config'],
                              python_batch=reference, c_batch=actual[gi]['batch']['items'][i],
                              python_single=expected_single[gi][i], c_single=actual[gi]['single'][i],
                              build_batch=compiled[gi]['batch']['items'][i], build_single=compiled[gi]['single'][i])
                (output / (source['uid'] + '.json')).write_text(json.dumps(detail, indent=2), encoding='utf-8')
                lines = [f"# {source['uid']}", '',
                         f"[完整输入参数及三方原始结果]({source['uid']}.json)", '']
                for mode in ('batch', 'single'):
                    sides = [detail[prefix + mode] for prefix in ('python_', 'build_', 'c_')]
                    lines += [f'## {mode}', '', '| 峰 | 字段 | Python 源码 | Cython build | C DLL |', '|---|---|---|---|---|']
                    for peak_index in range(max(len(side['peaks']) for side in sides)):
                        for key in ('a', 'b', 'c'):
                            values = [str(side['peaks'][peak_index][key]) if peak_index < len(side['peaks']) else 'missing' for side in sides]
                            lines.append('| ' + ' | '.join([str(peak_index + 1), key] + values) + ' |')
                    lines += ['', '| 项目 | Python 源码 | Cython build | C DLL |', '|---|---|---|---|']
                    for key in ('status', 'alerts'):
                        lines.append('| ' + key + ' | ' + ' | '.join(json.dumps(side[key], ensure_ascii=False) for side in sides) + ' |')
                    lines += ['| 峰数 | ' + ' | '.join(str(len(side['peaks'])) for side in sides) + ' |', '']
                (output / (source['uid'] + '.md')).write_text('\n'.join(lines), encoding='utf-8')
                try:
                    for mode in ('batch', 'single'):
                        for side in ('python', 'build', 'c'):
                            for peak in detail[side + '_' + mode]['peaks']:
                                if not peak['c'] > suite[gi]['config']['threshold']:
                                    raise AssertionError(f'{side}.{mode}: final c must exceed threshold: {peak}')
                    batch = compare(actual[gi]['batch']['items'][i], reference, 'batch')
                    single = compare(actual[gi]['single'][i], expected_single[gi][i], 'single')
                    build_batch = compare(compiled[gi]['batch']['items'][i], reference, 'build.batch')
                    build_single = compare(compiled[gi]['single'][i], expected_single[gi][i], 'build.single')
                    c_build_batch = compare(actual[gi]['batch']['items'][i], compiled[gi]['batch']['items'][i], 'c_vs_build.batch')
                    c_build_single = compare(actual[gi]['single'][i], compiled[gi]['single'][i], 'c_vs_build.single')
                except AssertionError as error:
                    record['error'] = str(error)
                    raise
                record.update(passed=True, peaks=len(reference['peaks']),
                              max_absolute_delta=max(batch, single, build_batch, build_single, c_build_batch, c_build_single))
            setattr(ParityTests, 'test_' + item['uid'], test)
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(ParityTests)
    if tests.countTestCases() < 100: raise AssertionError('At least 100 cases required')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    peak_cases = sum(bool(item['peaks']) for group in expected for item in group['items'])
    report = dict(input_directory=str(args.inputs.resolve()), cases=result.testsRun, comparisons=result.testsRun * 6,
                  implementations=['python_source', 'cython_build', 'c_dll'], compiled_modules=compiled_payload['modules'],
                  passed=result.testsRun - len(result.failures) - len(result.errors),
                  peak_cases=peak_cases, max_absolute_delta=max((r.get('max_absolute_delta', 0) for r in records), default=0),
                  model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  abs_tolerance=1e-12, rel_tolerance=1e-10, records=records)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    from render_report import render
    render(output)
    rows = ['# Python / Cython build / C DLL 三方推理对照报告', '',
            f"通过 {report['passed']}/{report['cases']}；返回峰的用例 {peak_cases}；最大差异 {report['max_absolute_delta']}", '',
            '| 用例（点击查看三方结果） | 通过 | 峰数 | 最大差异 |', '|---|---|---|---|']
    rows += [f"| [{r['uid']}]({r['uid']}.md) | {r['passed']} | {r.get('peaks', '-')} | {r.get('error', r.get('max_absolute_delta', '-'))} |" for r in records]
    (output / 'README.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    # Prevent a vacuous pass if no meaningful peak-producing inputs were exercised.
    if peak_cases < 50: raise AssertionError(f'Insufficient peak-producing cases: {peak_cases}')
    print(f'Report: {output / "report.json"}', flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
