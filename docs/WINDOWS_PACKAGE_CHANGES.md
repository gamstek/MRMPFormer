# Windows 集成包修改点声明

## 对比基线

本说明相对于以下旧集成目录于 2026-09-21 的文件快照：

```text
C:\Users\lengjing\workspace\code\ms8000server\TQ8000Server\third_party\mrmpformer\runtime\windows
```

旧 `mrmpformer.dll` SHA256：
`80c9f8e7adeb72bf3791476e0beae9aad276bf76e7e210331c574303645e6dab`

旧 `mrmpformer.h` SHA256：
`8559132b9c5f3c6d5a671eb5942a5aa2d176350bc591c3c40d68f44113afe010`

此基线是原 MRMPFormer 包，不是更早的 QuanFormer 接口。

## 调用接口与模型

- 去除注释、空白和 BOM 后，两份公开头文件声明一致：函数签名、结构体字段类型与顺序、枚举数值未变。原有 C 调用结构可以沿用。
- 新头文件补充了参数、生命周期和错误处理说明。使用新包配套的头文件、导入库和 DLL；接口布局一致不代表推理行为和默认值不变。
- ONNX 模型文件未变，两份 `mrmpformerv2.onnx` 的 SHA256 均为：
  `d0b612a786fc937ee44170a4e60e59a59d23fe7acdc2b85f5df302ae2dbee039`。

## 默认参数变化

旧值已通过调用旧 DLL 的 `qf_default_config()` 核实。显式传入的参数不会自动改成新默认值。

| 参数 | 旧包默认值 | 新包默认值 | 集成影响 |
|---|---|---|---|
| `threshold` | 0.99 | 0.5 | 影响峰是否保留；请按业务需要显式设置 |
| `smooth_sigma` | 0 | 0.8 | 默认启用高斯平滑；调用方传入未平滑原始强度，避免重复平滑 |
| `use_gpu` | -1，CPU | 0，GPU 优先、失败回退 CPU | 用 `qf_is_gpu_enabled()` 查询实际设备 |

`max_workers=0`、`task_timeout_sec=300`、`batch_size=128`、`min_chrom_points=10`、`min_max_intensity=1000` 保持不变。`use_gpu=-1` 仍强制 CPU，`1` 仍要求 GPU；无法启用 GPU 时必须报错。

## 当前推理行为

- 新包通过私有 CPython 和 Cython 模块调用共用的 Python MassNova 前后处理，模型仍由 ONNX Runtime 执行，减少 C 与 Python 分别实现算法造成的差异。
- **所有返回峰必须满足 `c > threshold`**，等于阈值也会被过滤。模型峰的 c 为模型置信度，信号兜底峰的 c 为信号规则分数；两者都接受最终阈值过滤。信号分数不是模型概率。
- 同通道、同范围的窗口复用推理结果；同一模型框最多分配给一个候选。最终 a/b 边界在 `1e-6` 分钟容差内相同的框只保留一个，模型来源优先，同来源保留高分框。
- 不同边界的重叠峰仍使用峰顶距离、重叠比例和深谷判断，避免误合并真实相邻峰。
- 返回结构仍为 `uid/status/peaks/alerts`，每个峰使用 `a/b/c`。最终无峰时返回 `alert` 和 `NO_PEAK_FOUND`；通过质量检查且存在合格峰时返回 `ok`。

本次已验证的修正示例：`test3_3_016` 在 threshold=0.8 时由 4 框变为 3 框；`test3_3_011` 在 threshold=0.5 时不再返回 c=0.3932536 的信号峰。这是新实现修正前后的回归记录，不是对旧基线 DLL 全量推理结果的比较结论。

## 交付文件和部署变化

旧目录只有 7 个文件，ONNX Runtime DLL 位于根目录。新包新增私有 Python、Cython 模块及其依赖，**必须整体部署并保留子目录结构**。

```text
mrmpformer.dll / mrmpformer.lib / mrmpformer.h
mrmpformerv2.onnx
python311.dll / python311.zip / python311._pth
python/                         私有模块与运行依赖
  Lib/site-packages/onnxruntime/capi/
    onnxruntime_providers_cuda.dll
    onnxruntime_providers_shared.dll
licenses/                       随包许可证
README_INTEGRATION.md            集成说明
CHANGELOG.md                     本修改点声明
```

根目录还包含配套 Python 与 MSVC 运行库文件，以实际交付目录为准。不要只替换 `mrmpformer.dll`，也不要把旧包根目录的 ONNX Runtime DLL 混入新包。

客户机器无需安装 Python、pip、Conda 或 PyTorch。同一包支持 CPU 和 CUDA；GPU 使用目标机兼容的 NVIDIA 驱动、CUDA 12 / cuDNN 9 运行环境。CUDA provider 已位于上述 Python 子目录，不是被删除了；包内不附带 NVIDIA pip CUDA 包。

## 验证范围

发布脚本检查独立目录下 C 接口运行、CPU/自动/强制 GPU 行为，并对 256 条真实输入执行源码 Python、Cython 和 C DLL 的批量/单条三方比较（1536 次），同时检查每个返回 c 严格大于请求阈值。失败则不发布新包。

详细报告与文件哈希保存在开发仓库 `tests/results/`，不随运行包交付。无可用 NVIDIA GPU 的构建机只能验证 CPU 和自动回退，不能替代 GPU 实机验收。旧基线目录本次仅进行了文件、接口和默认参数核对，未执行旧 DLL 的全量结果或性能对比。
