# -*- coding: utf-8 -*-
"""导出 MRMPFormer checkpoint 为 ONNX。

用法（在 model/ 目录下）：
    python -m tools.export_onnx --checkpoint checkpoint/mrmpformerv2.pth --out checkpoint/mrmpformerv2.onnx

ONNX 接口约定：
    输入:
      image    : float32 [B, 3, H, W]，RGB，0-255 原始像素（预处理已内置：
                 /255 + Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])，
                 与 utils/predict_utils.py 的 transform 完全一致）
      img_size : float32 [2] = (W, H)，用于把归一化框映射回像素坐标
    输出:
      scores      : float32 [B, Q]      各 query 的峰值概率（类别 0 的 softmax，与
                  predict_utils.predict 的 probas = softmax[...,:1] 口径一致）
      boxes_xyxy  : float32 [B, Q, 4]   像素坐标 (x1, y1, x2, y2)（与 rescale_bboxes 一致）
      boxes_norm  : float32 [B, Q, 4]   归一化 (cx, cy, w, h)（模型原始输出）

    B / H / W 均为动态维度（dynamic axes），Q 为固定 num_queries。
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch import nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # model/
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from framework.util.misc import NestedTensor  # noqa: E402
from models import build_model  # noqa: E402
from utils.torch_device import load_torch_checkpoint  # noqa: E402

# 与 utils/predict_utils.py 的 transform 保持一致
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class MRMPFormerOnnx(nn.Module):
    """ONNX 包装：内置预处理 + 后处理，输出像素坐标结果。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.aux_loss = False  # 推理不需要 aux_outputs，简化导出图
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor, img_size: torch.Tensor):
        # --- 预处理：等价 T.ToTensor() + T.Normalize(...) ---
        x = image / 255.0
        x = (x - self.mean) / self.std

        # 等尺寸 batch 的 NestedTensor：mask 全 False（无 padding）
        mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3],
                           dtype=torch.bool, device=x.device)
        out = self.model(NestedTensor(x, mask))

        logits = out["pred_logits"]          # [B, Q, 2]
        boxes = out["pred_boxes"]            # [B, Q, 4] 归一化 cxcywh
        scores = logits.softmax(-1)[..., :1].squeeze(-1)  # [B, Q] 峰概率

        # --- 归一化 cxcywh → 像素 xyxy（与 predict_utils.rescale_bboxes 同式）---
        scale = torch.cat([img_size, img_size], dim=0)    # [4] = (W,H,W,H)
        b = boxes * scale
        xy1 = b[..., :2] - b[..., 2:] / 2                  # cx,cy -> x1,y1
        xy2 = xy1 + b[..., 2:]                             # x2 = x1 + w, y2 = y1 + h
        boxes_xyxy = torch.cat([xy1, xy2], dim=-1)         # [B, Q, 4]
        return scores, boxes_xyxy, boxes


def load_model(checkpoint_path: str) -> nn.Module:
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    train_args = checkpoint.get("args", None) if isinstance(checkpoint, dict) else None
    if train_args is None:
        raise RuntimeError("checkpoint 内无 args，无法重建模型结构")
    train_args.device = "cpu"
    if not getattr(train_args, "model", None):
        train_args.model = "quanformer"

    result = build_model(train_args)
    model = result[0] if isinstance(result, tuple) else result

    report = model.load_state_dict(state_dict, strict=False)
    miss, unexp = list(report.missing_keys), list(report.unexpected_keys)
    if miss or unexp:
        print(f"[WARN] 权重加载差异：missing={len(miss)} unexpected={len(unexp)}")
        print(f"[WARN]   missing 前 10: {miss[:10]}")
        print(f"[WARN]   unexpected 前 10: {unexp[:10]}")
    else:
        print("[INFO] 模型权重完整加载（strict 匹配）")
    model.eval()
    return model


def export(checkpoint_path: str, out_path: str, opset: int, example_hw=(256, 320)):
    model = load_model(checkpoint_path)
    wrapper = MRMPFormerOnnx(model).eval()

    h, w = example_hw
    dummy_image = torch.rand(1, 3, h, w, dtype=torch.float32) * 255.0
    dummy_size = torch.tensor([float(w), float(h)], dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_size),
            out_path,
            input_names=["image", "img_size"],
            output_names=["scores", "boxes_xyxy", "boxes_norm"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "img_size": {},
                "scores": {0: "batch"},
                "boxes_xyxy": {0: "batch"},
                "boxes_norm": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[INFO] ONNX 已导出: {out_path} ({size_mb:.1f} MB, opset {opset})")
    return out_path


def verify(checkpoint_path: str, onnx_path: str):
    """Torch vs ONNX Runtime 数值一致性 + 动态尺寸验证。"""
    import onnxruntime as ort

    model = MRMPFormerOnnx(load_model(checkpoint_path)).eval()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    rng = np.random.RandomState(0)
    for (h, w) in [(256, 320), (192, 448)]:  # 两种尺寸验证动态轴
        img = rng.rand(1, 3, h, w).astype(np.float32) * 255.0
        size = np.array([w, h], dtype=np.float32)
        with torch.no_grad():
            t_scores, t_xyxy, t_norm = model(
                torch.from_numpy(img), torch.from_numpy(size))
        o_scores, o_xyxy, o_norm = sess.run(
            None, {"image": img, "img_size": size})
        d1 = np.abs(t_scores.numpy() - o_scores).max()
        d2 = np.abs(t_xyxy.numpy() - o_xyxy).max()
        d3 = np.abs(t_norm.numpy() - o_norm).max()
        print(f"[VERIFY] {w}x{h}: |Δscores|={d1:.2e} |Δxyxy|={d2:.2e} |Δnorm|={d3:.2e}")
        assert d1 < 1e-3 and d2 < 1e-2 and d3 < 1e-4, "ONNX 与 Torch 输出偏差过大"
    print("[VERIFY] ONNX 数值一致性通过（含动态尺寸）")
    print(f"[INFO] max score 示例: {o_scores.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="导出 MRMPFormer checkpoint 为 ONNX")
    parser.add_argument("--checkpoint", default="checkpoint/mrmpformerv2.pth")
    parser.add_argument("--out", default="checkpoint/mrmpformerv2.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    export(args.checkpoint, args.out, args.opset)
    if not args.no_verify:
        verify(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
