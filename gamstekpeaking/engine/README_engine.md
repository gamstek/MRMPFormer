# engine/ — 模型推理引擎（预留）

未来从 `model/quanformer/` 逐步迁移模型加载、推理、预测逻辑至此。

## 规划

| 子模块 | 预期功能 |
|--------|---------|
| `loader.py` | 模型权重安全加载（`safe_torch_load` 封装，兼容跨 PyTorch 版本） |
| `predictor.py` | DETR 推理管线（EIC 提取 + 模型预测 + 后处理） |
| `device.py` | 设备自动选择（CUDA > MPS > CPU 检测） |

## 设计原则

- **解耦**: `engine/` 不 import 任何 `pages/` 或 `workers/` 代码
- **统一接口**: 所有模型操作通过 `engine/` 暴露，GUI 只依赖 engine 的公共 API
- **渐进迁移**: 保持与现有 `model/quanformer/` 的兼容性，逐步替换而非一次性重写
- **设备无关**: 严禁硬编码 `device='cuda'`，使用自动检测
