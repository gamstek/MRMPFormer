"""Validate an ONNX model against the unchanged C++ tensor contract."""
import argparse
import json
from pathlib import Path

import numpy as np


def check(model_path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    assert [v.name for v in inputs] == ['image', 'img_size']
    assert [v.name for v in outputs] == ['scores', 'boxes_xyxy', 'boxes_norm']
    assert all(v.type == 'tensor(float)' for v in inputs + outputs)
    assert [len(v.shape) for v in inputs] == [4, 1]
    assert [len(v.shape) for v in outputs] == [2, 3, 3]
    assert inputs[0].shape[1] == 3 and inputs[1].shape == [2]
    assert outputs[1].shape[-1] == outputs[2].shape[-1] == 4
    static_queries = [v.shape[1] for v in outputs if isinstance(v.shape[1], int) and v.shape[1] > 0]
    assert len(set(static_queries)) <= 1
    query_count = None
    checks = []
    rng = np.random.default_rng(42)
    for batch, h, w in [(1, 300, 400), (2, 300, 400), (3, 192, 448)]:
        image = rng.uniform(0, 255, (batch, 3, h, w)).astype(np.float32)
        size = np.array([w, h], dtype=np.float32)
        scores, pixel, norm = session.run(None, {'image': image, 'img_size': size})
        if query_count is None:
            query_count = scores.shape[1]
        assert scores.shape == (batch, query_count)
        assert pixel.shape == norm.shape == (batch, query_count, 4)
        assert all(np.isfinite(v).all() for v in (scores, pixel, norm))
        assert ((scores >= 0) & (scores <= 1)).all()
        scale = np.array([w, h, w, h], dtype=np.float32)
        scaled = norm * scale
        expected = np.concatenate([scaled[..., :2] - scaled[..., 2:] / 2,
                                   scaled[..., :2] + scaled[..., 2:] / 2], axis=-1)
        error = float(np.max(np.abs(pixel - expected)))
        assert error < 1e-3, error
        assert (pixel[..., 2:] >= pixel[..., :2]).all()
        checks.append(dict(batch=batch, height=h, width=w,
                           scores_shape=list(scores.shape), boxes_shape=list(pixel.shape),
                           pixel_mapping_max_error=error))
    return dict(model=str(Path(model_path).resolve()), provider='CPUExecutionProvider',
                contract_compatible=True, checks=checks,
                scope='ONNX contract and runtime only; not a compiled C API integration test')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoint/mrmpformerv2.onnx')
    parser.add_argument('--report')
    args = parser.parse_args()
    report = check(args.model)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
