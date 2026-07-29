# -*- coding: utf-8 -*-
"""PyTorch 推理设备选择：优先 CUDA，不可用则回退 CPU。"""

import torch

_CACHED_DEVICE = None


def resolve_torch_device(verbose=True):
    """
    检测 CUDA 并返回 torch.device。
    verbose=True 时在控制台打印设备信息（仅首次解析时打印，避免批量重复刷屏）。
    """
    global _CACHED_DEVICE
    if _CACHED_DEVICE is not None:
        if verbose:
            _print_device_info(_CACHED_DEVICE, cached=True)
        return _CACHED_DEVICE

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    _CACHED_DEVICE = device
    if verbose:
        _print_device_info(device, cached=False)
    return device


def reset_torch_device_cache():
    """测试或切换环境时清空缓存。"""
    global _CACHED_DEVICE
    _CACHED_DEVICE = None


def load_torch_checkpoint(path, map_location=None):
    """
    加载本地 .pth 检查点（含 model/args 等完整对象）。
    PyTorch 2.6+ 默认 weights_only=True，需显式 False 才能读取 argparse.Namespace。
    """
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _print_device_info(device, cached=False):
    if cached:
        if device.type == "cuda":
            idx = torch.cuda.current_device()
            print(
                "[INFO] PyTorch 推理设备: CUDA — %s (cuda:%d，沿用本次进程)"
                % (torch.cuda.get_device_name(idx), idx)
            )
        else:
            print("[INFO] PyTorch 推理设备: CPU（沿用本次进程）")
        return
    prefix = "[INFO] PyTorch 推理设备"
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        print("%s: CUDA — %s (cuda:%d)" % (prefix, name, idx))
        print("[INFO] CUDA 计算能力: %d.%d" % (cap[0], cap[1]))
        if torch.version.cuda:
            print("[INFO] PyTorch 编译 CUDA 版本: %s" % torch.version.cuda)
        mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024.0 ** 3)
        print("[INFO] GPU 显存: %.2f GB" % mem_gb)
    else:
        print("%s: CPU" % prefix)
        if getattr(torch.version, "cuda", None) is None:
            print("[INFO] 未使用 GPU：当前安装的 PyTorch 为 CPU 版（无 CUDA 支持）")
        else:
            print(
                "[INFO] 未使用 GPU：torch.cuda.is_available()=False"
                "（请检查 NVIDIA 驱动、CUDA 与 PyTorch GPU 版是否匹配）"
            )
