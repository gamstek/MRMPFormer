"""Import real chromatograms from a test3 validation package; never synthesize signals."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def load_cases(directory):
    files = sorted(Path(directory).glob('*.json'))
    if len(files) < 100:
        raise ValueError('At least 100 input JSON files required. Import real cases with tests/import_inputs.py')
    groups = {}
    seen = set()
    for file in files:
        case = json.loads(file.read_text(encoding='utf-8'))
        uid = case['input']['uid']
        if uid != file.stem or uid in seen:
            raise ValueError(f'{file}: UID must be unique and match filename')
        seen.add(uid)
        key = json.dumps(case['config'], sort_keys=True)
        group = groups.setdefault(key, {'config': case['config'], 'items': []})
        group['items'].append(case['input'])
    return list(groups.values())


def import_cases(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    config = {key: manifest['config'][key] for key in (
        'threshold', 'smooth_sigma', 'use_gpu', 'batch_size', 'min_chrom_points', 'min_max_intensity')}
    for key in ('threshold', 'smooth_sigma', 'min_max_intensity'):
        config[key] = f32(config[key])
    cases = []
    for dataset in manifest['golden_sets']:
        relative = dataset['input']
        raw = (source / relative).read_bytes()
        payload = json.loads(raw)
        for index, original in enumerate(payload['items']):
            uid = f"{dataset['sample']}_{index:03d}"
            item = dict(original, uid=uid)
            for key in ('smooth_sigma', 'mzq1', 'mzq3'):
                if key in item:
                    item[key] = f32(item[key])
            cases.append(dict(config=config, input=item, source=dict(
                package=manifest['package'], sample=dataset['sample'], file=relative,
                item_index=index, original_uid=original['uid'],
                input_sha256=hashlib.sha256(raw).hexdigest(),
                model_sha256=manifest['model_sha256'])))
    if len(cases) < 100:
        raise ValueError('At least 100 real inputs required')
    if output.exists() and any(output.glob('*.json')):
        raise ValueError('Output already contains cases; use an empty directory')
    output.mkdir(parents=True, exist_ok=True)
    rows = ['# 真实输入数据', '',
            f"来源：`{manifest['package']}`，{len(manifest['golden_sets'])} 个样本，{len(cases)} 条分层验收输入。", '',
            '保留完整原始 x/y，无平滑、重采样或合成。UID 加样本前缀，原 UID 与来源 SHA256 记录在 source 中。',
            '配置来自源 manifest；C ABI 的 float 参数统一转换为 float32。', '',
            '| 文件 | 样本 | 原 UID | 点数 |', '|---|---|---|---|']
    for case in cases:
        uid = case['input']['uid']
        (output / (uid + '.json')).write_text(json.dumps(case, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        rows.append(f"| [{uid}]({uid}.json) | {case['source']['sample']} | {case['source']['original_uid']} | {len(case['input']['x'])} |")
    (output / 'README.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(f'Imported {len(cases)} real cases into {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Root of the real validation package')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'inputs')
    args = parser.parse_args()
    import_cases(args.source, args.output)
