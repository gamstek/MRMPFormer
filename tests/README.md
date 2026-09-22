# Python / Cython build / C DLL 三方推理对比

`inputs/` 包含 256 条真实 MRM 输入，来自 `test3_cpp_validation_1d4ba0f/golden/inputs/` 的四个分层验收集（test3_1、test3_3、test3_7、test3_56，各 64 条）。源包另有 58 个完整样本，本套回归使用其中已选定的验收子集。

每个 JSON 包含配置、完整原始 x/y、样本名、原 UID、来源文件与 SHA256。原始 x/y 不平滑、不重采样；文件 UID 使用样本名和索引避免重名。C ABI 的 float 参数统一为 float32。`inputs/README.md` 是逐例索引。

## 运行

真实输入 JSON 不提交到 Git。首次检出后，先用下方导入命令将验收包导入 `tests/inputs/`，再运行对比或一键打包。

在仓库根目录使用 `.venv`：

```powershell
uv run tests/run_parity.py
```

需要已生成的 `build/windows/`、`model/build/python/` 中的 Cython 扩展、CMake 和 MSVC。运行时读取现有 JSON，不会重新生成输入。可用 `--inputs`、`--package`、`--output` 指定目录。

三方分别使用 Python `.py` 源码、独立进程加载的四个 Cython `.pyd`、原生进程调用的已打包 C DLL。批量和单条调用各自两两比较，共 1536 次对照。检查状态、告警、峰数量、顺序和逐峰 a/b/c；容差 `atol=1e-12, rtol=1e-10`，拒绝 NaN/Inf，至少 50 例须返回峰。此处验证三方实现一致性，不将源包历史 golden 输出视作当前结果，也不评估检测准确率。

## 完整报告

`results/report.html` 自动生成，包含所有输入参数、采样点和三方批量/单条完整输出。逐例 JSON/Markdown、`actual.json`、`compiled.json`、`expected.json` 与汇总 `report.json` 同时保存。仅重建 HTML 可执行 `uv run tests/render_report.py`。

## 数据导入和自测

```powershell
uv run tests/import_inputs.py C:\Users\lengjing\Downloads\test3_cpp_validation_1d4ba0f --output <空目录>
uv run python -m unittest discover -s tests -p "test_*.py"
uv run tests/check_devices.py
```

导入器只读取真实数据，不覆盖已有用例。`check_devices.py` 使用真实色谱验证 CPU、自动选择和强制 GPU；无可用 CUDA 环境时只验证 CPU 与回退。设备策略、依赖解析等单元测试保留必要的模拟对象，它们不充当推理数据。

测试驱动源文件统一在 `tests/`，编译产物在 `tests/build/`，报告在 `tests/results/`。一键打包也会执行这里的三方对比，失败则不发布新包。
