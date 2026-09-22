"""Verify CPU, automatic selection, and required-GPU behavior through the packaged C ABI."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check(package, output):
    package = package.resolve()
    case = json.loads((ROOT / 'tests/inputs/test3_1_000.json').read_text(encoding='utf-8'))
    report = {}
    with tempfile.TemporaryDirectory(prefix='mrmpformer-devices-') as folder:
        work = Path(folder)
        shutil.copy2(ROOT / 'tests/build/Release/inference_parity_runner.exe', work / 'runner.exe')
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith(('PYTHON', 'CONDA', 'VIRTUAL_ENV', 'MRMPFORMER'))}
        # Unlike the CPU isolation test, GPU validation needs the target's CUDA/cuDNN paths.
        env['PATH'] = str(package) + os.pathsep + os.environ.get('PATH', '')
        env['MPLCONFIGDIR'] = str(work / 'matplotlib')
        env['PYTHONHOME'] = str(work / 'nonexistent')
        for mode, name in [(-1, 'cpu'), (0, 'auto'), (1, 'required_gpu')]:
            config = dict(case['config'], use_gpu=mode)
            (work / 'input.json').write_text(json.dumps([dict(config=config, items=[case['input']])]), encoding='utf-8')
            result = subprocess.run([str(work / 'runner.exe'), str(package / 'mrmpformerv2.onnx'), 'input.json', 'output.json'],
                                    cwd=work, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
            if result.returncode:
                if mode != 1 or 'CUDA' not in result.stderr:
                    raise RuntimeError(result.stdout + result.stderr)
                report[name] = dict(available=False, error=result.stderr)
                continue
            actual = json.loads((work / 'output.json').read_text(encoding='utf-8'))[0]
            if mode == -1 and actual['gpu_enabled']:
                raise AssertionError('Forced CPU unexpectedly selected GPU')
            if mode == 1 and not actual['gpu_enabled']:
                raise AssertionError('Required GPU silently fell back')
            peaks = actual['batch']['items'][0]['peaks']
            if not peaks:
                raise AssertionError(f'{name}: expected peaks for the real chromatogram')
            if mode == 0 and not actual['gpu_enabled'] and actual['batch'] != report['cpu']['result']['batch']:
                raise AssertionError('Automatic CPU fallback differs from forced CPU')
            report[name] = dict(available=True, gpu_enabled=actual['gpu_enabled'], result=actual,
                                diagnostics=result.stdout + result.stderr)
    if report['auto']['gpu_enabled'] != report['required_gpu']['available']:
        raise AssertionError('Automatic and required GPU availability disagree')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Device checks: CPU passed, auto GPU={report["auto"]["gpu_enabled"]}; {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT / 'build/windows')
    parser.add_argument('--output', type=Path, default=ROOT / 'tests/results/devices.json')
    args = parser.parse_args()
    check(args.package, args.output)
