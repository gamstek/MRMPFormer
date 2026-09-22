"""Run model/build Cython extensions directly, in an independent Python process."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build = ROOT / 'model/build/python'
    sys.path[:0] = [str(build), str(ROOT / 'model')]
    from inference import massnova_bridge_native as bridge
    from inference import massnova_runtime, massnova, onnx_window_predictor
    modules = (bridge, massnova_runtime, massnova, onnx_window_predictor)
    paths = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        if path.suffix != '.pyd' or not path.is_relative_to(build.resolve()):
            raise RuntimeError(f'Expected compiled model/build extension, got {path}')
        paths[module.__name__] = str(path)
    output = []
    suite = json.loads(args.inputs.read_text(encoding='utf-8'))
    for i, group in enumerate(suite):
        print(f'Running compiled Cython {i + 1}/{len(suite)}...', flush=True)
        bridge.initialize(str(args.model), json.dumps(group['config']))
        try:
            batch = json.loads(bridge.process_json(json.dumps({'items': group['items']})))
            single = [json.loads(bridge.process_json(json.dumps({'items': [item]})))['items'][0]
                      for item in group['items']]
            output.append(dict(batch=batch, single=single))
        finally:
            bridge.shutdown()
    args.output.write_text(json.dumps({'modules': paths, 'groups': output}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
